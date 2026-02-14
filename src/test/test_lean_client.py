from custom_tools.lean_client import LSPClient

def assert_false(client):
    client.shutdown()
    assert(False)

client = LSPClient()

path_to_test_files = "./lean_files/"

correct, _ = client.check_file_with_nice_output(path_to_test_files + "test_1.lean")
if not correct:
    assert_false(client)
correct, _ = client.check_file_with_nice_output(path_to_test_files + "test_2.lean")
if not correct:
    assert_false(client)

for test_file in ("test_3.lean", "test_4.lean") :
    # These files are syntatically correct but are not valid proofs
    correct, _ = client.check_file_with_nice_output(path_to_test_files + test_file, insist_on_theorem = False)
    if not correct:
        assert_false(client)
    correct, _ = client.check_file_with_nice_output(path_to_test_files + test_file, insist_on_theorem = True)
    if correct:
        assert_false(client)

correct, _ = client.check_file_with_nice_output(path_to_test_files + "test_5.lean")
if correct:
    assert_false(client)

correct, _ = client.check_file_with_nice_output(path_to_test_files + "test_6.lean")
if not correct:
    assert_false(client)

correct, _ = client.check_file_with_nice_output(path_to_test_files + "test_7.lean")
if correct:
    assert_false(client)

correct, _ = client.check_file_with_nice_output(path_to_test_files + "test_8.lean")
if correct:
    assert_false(client)

client.shutdown()
print("\nLean Client passed all tests")
