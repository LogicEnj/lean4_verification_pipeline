import json
import os
import subprocess
import select
import time

def package_to_path(package):
    return os.path.join(os.path.dirname(__file__) , "../../lean_project/.lake/packages/" + package + "/.lake/build/lib")

def setup_env():
    os.environ["LEAN_SERVER_LOG_DIR"] = "./logs"
    os.environ["LD_LIBRARY_PATH"] = ""
    os.environ["LEAN_COMMAND"] = os.environ["HOME"] + "/.elan/toolchains/leanprover--lean4---v4.17.0-rc1/bin/lean"
    packages = ['Cli', 'batteries', 'Qq', 'aesop', 'proofwidgets', 'importGraph', 'LeanSearchClient', 'plausible']
    os.environ["LEAN_PATH"] = os.path.join(os.path.dirname(__file__), "../../lean_project/mathlib4-old/.lake/build/lib") + ":" + ":".join([package_to_path(p) for p in packages])

def contains_error(diagnostics) :
    return any(entry['severity'] <= 1 and entry.get('message') != 'no goals to be solved' for entry in diagnostics)

class LSPClient:
    def __init__(self, read_response_timeout = 60, general_timeout = 60):
        """Start Lean 4 LSP server"""
        setup_env()
        self.read_response_timeout = read_response_timeout
        self.general_timeout = general_timeout
        self.message_id = 1
        self.process = subprocess.Popen(
            [os.environ["LEAN_COMMAND"], "--server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # Line buffered
        )
        time.sleep(0.5)
        # Wait for initialization response
        init_id = self.send_message("initialize", {
            "processId": self.process.pid,
            "rootUri": None,
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": True,
                        "change": 2
                    }
                }
            }
        })
        while True:
            response = self.read_response()
            if not response:
                continue  # Skip null responses
            if response.get("id") == init_id:
                break
            else:
                print(f"Ignoring message: {response.get('method')}")
        self.send_notification("initialized", {})

    def send_notification(self, method, params):
        """Send LSP notification (no ID)"""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        }
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        self.process.stdin.write((header + content).encode('utf-8'))
        self.process.stdin.flush()
        
    def send_message(self, method, params):
        """Send LSP message with proper framing"""
        message = {
            "jsonrpc": "2.0",
            "id": self.message_id,
            "method": method,
            "params": params
        }
        self.message_id += 1
        
        content = json.dumps(message)
        header = f"Content-Length: {len(content)}\r\n\r\n"
        
        # Write header + content to stdin
        self.process.stdin.write((header + content).encode('utf-8'))
        self.process.stdin.flush()
        return message["id"]

    def read_response(self):
        """Robust LSP message reader with proper framing"""
        timeout = self.read_response_timeout
        try:
            start_time = time.time()
            content_length = 0
            headers_complete = False
            content = ''

            # Read headers until empty line
            while time.time() - start_time < timeout:
                rlist, _, _, = select.select([self.process.stdout], [], [], timeout)
                if rlist:
                    line = self.process.stdout.readline().strip()
                else:
                    raise TimeoutError(f"No output after {timeout} seconds")
                if not line:  # Empty line indicates end of headers
                    headers_complete = True
                    break
                if line.startswith(b"Content-Length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
            
            if content_length <= 0:
                print('Content Length <= 0, returning None')
                return None
            
            if not headers_complete :
                content_length += 2 

            # Read EXACTLY content_length bytes (as binary)
            content_bytes = b''
            remaining_len = content_length
            while remaining_len:
                content_bytes += self.process.stdout.read(remaining_len)
                remaining_len = content_length - len(content_bytes)
            
            # Decode with error handling
            try:
                content = content_bytes.decode('utf-8')
                return json.loads(content)
            except UnicodeDecodeError:
                # Fallback for malformed Unicode
                content = content_bytes.decode('utf-8', errors='replace')
                print(f"Unicode decode warning: {content[:200]}...")
                return None
            
        except Exception as e:
            print(f"JSON error: {str(e)}")
            print("Content length:", content_length)
            print('Full content:', content)
            return None

    def verify_lean_code(self, lean_code, file_uri):
        """Send a theorem and proof to Lean for verification"""
        timeout = self.general_timeout
        
        # Send textDocument/didOpen notification
        self.send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": file_uri,
                "languageId": "lean4",
                "version": 1,
                "text": lean_code
            }
        })

        # Get diagnostics with timeout
        start_time = time.time()
        diagnostics = []
        no_goals = []
        ret = True
        is_timeout = True
        empty_process = False
        
        while time.time() - start_time < timeout:  # Increased timeout
            response = self.read_response()
            if not response:
                continue  # Skip null responses

            if response.get("method") == "$/lean/fileProgress" and response['params']['textDocument']['uri'] == file_uri:
                if response['params']['processing']:
                    if response['params']['processing'][0]['kind'] == 2 :
                        is_timeout = False
                        break
                    empty_process = False
                else:
                    empty_process = True
                continue

            if empty_process and response.get("method") == "workspace/semanticTokens/refresh":
                is_timeout = False
                break

            # Handle diagnostics
            if response.get("method") == "textDocument/publishDiagnostics" and response['params']['uri'] == file_uri:
                diagnostics = response["params"]["diagnostics"]
                if diagnostics:
                    no_goals = diagnostics
                    if contains_error(diagnostics) :
                        ret = False
                        is_timeout = False
                        break

            # Handle other messages
            elif response.get("id"):
                # Track pending requests if needed
                pass
        
        self.send_notification("textDocument/didClose", {
            "textDocument": {
                "uri": file_uri
            }
        })

        try:
            if is_timeout:
                raise TimeoutError(f"No output after {timeout} seconds")
            else:
                if not diagnostics:
                    diagnostics = no_goals
                return ret, diagnostics
        except Exception as e:
            print(str(e))
            return None, f"Exception in client: {str(e)}"

    def shutdown(self):
        """Properly shutdown the LSP server"""
        shutdown_id = self.send_message("shutdown", None)
        while True:
            response = self.read_response()
            if response.get("id") == shutdown_id:
                break
        self.send_message("exit", None)
        self.process.terminate()

    def check_text(self, text, file_uri = None):
        """High-level function to verify a theorem"""
        if file_uri is None:
            file_uri = text
        verified, result = self.verify_lean_code(text, file_uri)
        if verified is None:
            return False, result
        else:
            error_message = "\n".join(
                f"Line {error['range']['start']['line'] + 1}: {error['message']}"
                for error in result
            )
        return verified, error_message

    def check_file(self, path_to_file, insist_on_theorem : bool):
        """High-level function to verify a theorem"""
        lean_code = ""

        with open(path_to_file, 'r', encoding = 'utf-8') as file:
            lean_code = file.read()

        print(lean_code)

        if insist_on_theorem and not "theorem" in lean_code :
            return False, "No theorems in a file"
        else:
            return self.check_text(lean_code, path_to_file)

    def check_file_with_nice_output(self, path_to_file, insist_on_theorem : bool = False) :
        print('Verifying ', path_to_file)
        verified, error_message = self.check_file(path_to_file, insist_on_theorem)
        if verified:
            print("✅ Theorem verified successfully!")
        else:
            print("❌ Errors found in proof:")
            print(error_message)
        return verified, error_message
