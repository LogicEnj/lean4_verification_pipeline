from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.lean_client import LSPClient
from correct_funcs import find_all_symb_outside_braces, find_symb_outside_braces, rewrite_statement_with_names, write_full_statement
import pandas as pd
import os
import re

def del_absolete_hypotheses(client, statement, file_uri):
    passphrases = ['unnecessary have', 'All linting checks passed!', 'unused variable', 'no goals to be solved', 'synTaut']
    def get_lint_messages(diag):
        for item in diag:
            message = item.get('message')
            if message and any(phrase in message for phrase in passphrases):
                yield message

    def get_unused_names(messages):
        for message in messages:
            message = message.split('\n', 1)[0]
            match = re.search(r'unused variable `(.*)`', message)
            if match:
                yield match[1]

    def check_lint(version, statement, file_uri, client):
        client.send_notification("textDocument/didChange", {
            "textDocument": {
                "uri": file_uri,
                "version": version
            },
            "contentChanges": [{
                "text": statement
            }]
        })
        messages = None
        params = None
        while not messages:
            response = client.read_response()
            if response and response.get("method") == "textDocument/publishDiagnostics":
                params = response['params']
                if params['uri'] == file_uri and params['version'] == version:
                    messages = list(get_lint_messages(params['diagnostics']))
        return messages, params['diagnostics'], version + 1

    statement = re.sub(r':=\s*by', r':= by', statement) + '\n  #lint'
    statement = statement.replace('\ndef ', '\n/-- -/\ndef ')
    unnamed_ind = 0
    unnamed_pat = re.compile(r'(\n *have) *([:({])', flags = re.M)
    unnamed_have = unnamed_pat.search(statement)
    while unnamed_have:
        statement = statement[:unnamed_have.start()] + unnamed_have[1] + f' unnamed_{unnamed_ind} ' + unnamed_have[2] + statement[unnamed_have.end():]
        unnamed_ind += 1
        unnamed_have = unnamed_pat.search(statement)

    client.send_notification("textDocument/didOpen", {
        "textDocument": {
            "uri": file_uri,
            "languageId": "lean4",
            "version": 1,
            "text": statement
        }
    })
    messages, diag, version = check_lint(1, statement, file_uri, client)

    positions = ('start', 'end')
    elements = ('line', 'character')
    while 'no goals to be solved' in messages:
        txt_range = diag[0]['range']
        line0, pos0, line1, pos1 = tuple(txt_range[pos][el] for pos in positions for el in elements)
        txt_lines = statement.split('\n')
        line = txt_lines[line0][: pos0] + txt_lines[line1][pos1 :]
        for i in range(line1, line0, -1):
            txt_lines.pop(i)
        if line.strip():
            txt_lines[line0] = line
        else:
            txt_lines.pop(line0)
        statement = '\n'.join(txt_lines)
        messages, diag, version = check_lint(version, statement, file_uri, client)

    if 'synTaut' in messages[0]:
        statement = re.sub(r':=.*\n  #lint', r':= by rfl\n  #lint', statement, flags = re.M | re.DOTALL)
        messages, _, version = check_lint(version, statement, file_uri, client)

    haves = re.findall(r'unnecessary have (\S+)', messages[0])
    while haves:
        for have in haves:
            have = re.escape(have)
            find = re.search(rf'(\s*)have\s+{have}[:\s].*:= ', statement, flags = re.M | re.DOTALL)
            if not find:
                continue
            line_indent = len(find[1])
            start = find.start()
            end = start + find[0].find(':= ') + 3
            end += statement[end:].find('\n')
            lines = re.findall(r'\n[^\n]*', statement[end:], flags = re.M)
            ind = 0
            for i, line in enumerate(lines):
                if len(re.search('^\s*', line, flags = re.M)[0]) <= line_indent:
                    ind = i
                    break
            statement = statement[: start] + ''.join(lines[ind :])
        messages, _, version = check_lint(version, statement, file_uri, client)
        haves = re.findall(r'unnecessary have (\S+)', messages[0])

    client.send_notification("textDocument/didClose", {
        "textDocument": {
            "uri": file_uri
        }
    })
    statement = statement[: statement.rfind('\n')].replace('\n/-- -/\ndef ', '\ndef ')
    return list(get_unused_names(messages)), statement

def get_lemma_indices(path):
    filename_pattern = re.compile(r'^thm_(\d+)_(\d+)\.lean$')
    def _lemma_indices_(path):
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                match = filename_pattern.fullmatch(filename)
                if match:
                    task_number = int(match[1])
                    lemma_number = int(match[2])
                    yield task_number, lemma_number

    return sorted(list(_lemma_indices_(path)))

def get_fact_indices(path):
    filename_pattern = re.compile(r'^fact_(\d+)_(\d+)_(\d+)\.lean$')
    def _fact_indices_(path):
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                match = filename_pattern.fullmatch(filename)
                if match:
                    task_number = int(match[1])
                    lemma_number = int(match[2])
                    fact_number = int(match[3])
                    yield task_number, lemma_number, fact_number

    return sorted(list(_fact_indices_(path)))

def add_exists_as_var(conclusion, index): #conclusion starts with ∃
    pos = conclusion.find(',')
    pos2 = conclusion.find(':')
    if pos2 > -1 and pos2 < pos:
        pos = pos2
    variables = re.findall('\S+', conclusion[1 : pos])
    result = ''
    link = f'L{index}\n'
    for var in variables:
        result += f'  obtain ⟨{var}, h{var}_{index}⟩ := ' + link
        link = f'h{var}_{index}\n'
    return result

input_dir = get_input_file_or_folder('./config.yaml')
output_dir = get_output_file_or_folder('./config.yaml')
os.makedirs(output_dir, exist_ok=True)

print('input_dir:', input_dir)
print('output:', output_dir)

theorems = pd.read_csv(os.path.join(input_dir, 'theorems.csv'))

lemma_indices = get_lemma_indices(input_dir)
fact_indices = get_fact_indices(input_dir)

tactics = ['show_term  tauto', 'show_term solve_by_elim (maxDepth := 20)']
len_tactics = len(tactics)
lemma_lookup = re.compile(r'theorem.*:=', flags = re.M | re.DOTALL)

client = LSPClient()
for task_index, row in theorems.iterrows():
    print('Task', task_index)
    if not row['correct']:
        for i in range(len_tactics):
            os.system('touch ' + os.path.join(output_dir, f'task-{task_index}-{i}.lean'))
        continue
    if row['equivalence']:
        theorem = row['converse']
        proof = row['header'] + theorem + '  intro h\n'
        theorem_match = re.search(r'theorem[^{(:]*', theorem)
        begin_theorem_stat = theorem_match.end()
        end_theorem_stat = lemma_lookup.search(theorem)[0].find(':=') + theorem_match.start()
        _, _, th_conclusion = rewrite_statement_with_names(theorem[begin_theorem_stat : end_theorem_stat])
        implication = find_symb_outside_braces('→', th_conclusion)
        left = th_conclusion[: implication].strip()[1 : -1]
        num_cases = len(find_all_symb_outside_braces('∨', left)) - 1
        if num_cases > 1:
            proof += '  rcases h with' + ' |'.join(f' h{i}' for i in range(num_cases)) + '\n  <;>simp_all\n  <;>try norm_num'
        else:
            proof += '  simp_all\n  try norm_num'
        with open(os.path.join(output_dir, f'task-{task_index}-inv.lean'), 'w', encoding = 'utf-8') as proof_file:
            proof_file.write(proof)

    if not row['checked'].endswith('TRUE'):
        for i in range(len_tactics):
            os.system('touch ' + os.path.join(output_dir, f'task-{task_index}-{i}.lean'))
        continue

    proof = row['header'] + row['direct']

    local_lemma_indices = [n_lemma for n_task, n_lemma in lemma_indices if n_task == task_index]
    for lemma_index in local_lemma_indices:
        local_fact_indices = [n_fact for n_task, n_lemma, n_fact in fact_indices if n_task == task_index and n_lemma == lemma_index]
        fact = 0
        for fact_index in local_fact_indices:
            print(f'Fact {lemma_index}_{fact_index}')
            fact_filename = f'fact_{task_index}_{lemma_index}_{fact_index}.lean'
            with open(os.path.join(input_dir, fact_filename), 'r', encoding = 'utf-8') as fact_file:
                fact_stat = fact_file.read()
            unused_names, fact_stat = del_absolete_hypotheses(client, fact_stat, fact_filename)
            find = lemma_lookup.search(fact_stat)
            line = find[0]
            find_stat = re.search(r'theorem[^{(:]*', line)
            start_ind = find_stat.end()
            end_ind = line.find(':=')
            line = line[start_ind : end_ind].strip()
            intro_vars, hypotheses_with_names, conclusion = rewrite_statement_with_names(line)
            hypotheses_with_names = [(name, hypothesis) for name, hypothesis in hypotheses_with_names if name not in unused_names]
            fact_stat = fact_stat[: start_ind + find.start()] + ' ' + ' '.join(intro_vars) + ' ' + ' '.join(f'({name}: {hypothesis})' for name, hypothesis in hypotheses_with_names) + ': '  + conclusion + fact_stat[end_ind + find.start() :]
            with open(os.path.join(output_dir, fact_filename), 'w', encoding = 'utf-8') as fact_file:
                fact_file.write(fact_stat)
            hypotheses = [hypothesis for name, hypothesis in hypotheses_with_names]
            str_index = f'f{lemma_index}_{fact}'
            fact += 1
            if conclusion.startswith('∃'):
                proof += f'  have F{str_index[1: ]}: {conclusion} := by\n'
                proof += f'    have {str_index} ' + write_full_statement(intro_vars, hypotheses, conclusion, str_index) + ' := by sorry\n'
                proof += f'    show_term solve_by_elim (maxDepth := 20)\n'
                proof += add_exists_as_var(conclusion, str_index)
            else:
                proof += f'  have {str_index} ' + write_full_statement(intro_vars, hypotheses, conclusion, str_index) + ' := by sorry\n'

        print(f'Lemma {lemma_index}')
        lemma_filename = f'thm_{task_index}_{lemma_index}.lean'
        with open(os.path.join(input_dir, lemma_filename), 'r', encoding = 'utf-8') as lemma_file:
            lemma_stat = lemma_file.read()
        unused_names, lemma_stat = del_absolete_hypotheses(client, lemma_stat, lemma_filename)
        find = lemma_lookup.search(lemma_stat)
        line = find[0]
        find_stat = re.search(r'theorem[^{(:]*', line)
        start_ind = find_stat.end()
        end_ind = line.find(':=')
        line = line[start_ind : end_ind].strip()
        intro_vars, hypotheses_with_names, conclusion = rewrite_statement_with_names(line)
        hypotheses_with_names = [(name, hypothesis) for name, hypothesis in hypotheses_with_names if name not in unused_names]
        lemma_stat = lemma_stat[: start_ind + find.start()] + ' ' + ' '.join(intro_vars) + ' ' + ' '.join(f'({name}: {hypothesis})' for name, hypothesis in hypotheses_with_names) + ': ' + conclusion  + lemma_stat[end_ind + find.start() :]
        with open(os.path.join(output_dir, lemma_filename), 'w', encoding = 'utf-8') as lemma_file:
            lemma_file.write(lemma_stat)
        hypotheses = [hypothesis for name, hypothesis in hypotheses_with_names]
        if conclusion.startswith('∃'):
            proof += f'  have L{lemma_index}: {conclusion} := by\n'
            proof += f'    have l{lemma_index} ' + write_full_statement(intro_vars, hypotheses, conclusion, f'h{lemma_index}') + ' := by sorry\n'
            proof += f'    show_term solve_by_elim (maxDepth := 20)\n'
            proof += add_exists_as_var(conclusion, f'{lemma_index}')
        else:
            proof += f'  have l{lemma_index} ' + write_full_statement(intro_vars, hypotheses, conclusion, f'h{lemma_index}') + ' := by sorry\n'

    if ".num" not in proof and ".den" not in proof:
        proof = proof.replace("Rat", "Real")

    for ind, tac in enumerate(tactics):
        with open(os.path.join(output_dir, f'task-{task_index}-{ind}.lean'), 'w', encoding = 'utf-8') as proof_file:
            proof_file.write(proof + '  ' + tac)
client.shutdown()
