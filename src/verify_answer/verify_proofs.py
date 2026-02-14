import argparse
import os
from custom_tools.lean_client import LSPClient
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder

#######################################################################

input_dir = get_input_file_or_folder('./config.yaml')


print('Verifying', input_dir)


verification_result = []
client = LSPClient()

correct_files = []
incorrect_files = []

for filename in os.listdir(input_dir):
    full_path_to_file = os.path.join(input_dir, filename)
    if os.path.isfile(full_path_to_file):
        verify, _ = client.check_file_with_nice_output(full_path_to_file, insist_on_theorem = True)
        if verify :
            correct_files.append(filename)
        else:
            incorrect_files.append(filename)
        verification_result.append(verify)

client.shutdown()

print('Correct files:')
print('\n'.join(correct_files))

print('Incorrect files:')
print('\n'.join(incorrect_files))

from collections import Counter
count_dict = Counter(verification_result)
good = count_dict[True]
bad = count_dict[False]
total = good + bad
print(good, 'verified out of', total)


