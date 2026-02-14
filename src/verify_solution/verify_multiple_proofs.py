import os
import re
import numpy as np
from custom_tools.lean_client import LSPClient
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder

#######################################################################
def tasks(dirpath):
    def gen_tasks(dirpath):
        filename_pattern = re.compile(r'^task-(\d+)-.+\.lean$')
        for filename in os.listdir(dirpath):
            match = filename_pattern.fullmatch(filename)
            if match and os.path.isfile(os.path.join(dirpath, filename)):
                task_number = int(match.group(1))
                yield task_number, filename
    gen = list(gen_tasks(dirpath))
    filenames = list(filename for task_number, filename in gen)
    task_indices = sorted(list(set(task_number for task_number, filename in gen)))
    return task_indices, filenames

def local_tactic_indices(task_index, filenames):
    def gen_local(task_index, filenames):
        filename_pattern = re.compile(rf'^task-{task_index}-(\d+)\.lean$')
        for filename in filenames:
            match = filename_pattern.fullmatch(filename)
            if match:
                yield int(match.group(1))
    local_indices = sorted(list(gen_local(task_index, filenames)))
    return local_indices

input_dir = get_input_file_or_folder('./config.yaml')
print('Verifying', input_dir)

task_indices, filenames = tasks(input_dir)
have_equivalence = np.fromiter((f'task-{task_index}-inv.lean' in filenames for task_index in task_indices), dtype = bool)
good = np.zeros((have_equivalence.size, 2), dtype = bool)

client = LSPClient()
for ind, (task_index, equivalence) in enumerate(zip(task_indices, have_equivalence)):
    if equivalence:
        full_path_to_file = os.path.join(input_dir, f'task-{task_index}-inv.lean')
        verified, error_message = client.check_file_with_nice_output(full_path_to_file, insist_on_theorem = True)
        good[ind, 1] = verified
        full_path_to_error_message_file = full_path_to_file[:-5] + '.err'
        with open(full_path_to_error_message_file, "w", encoding = "utf-8") as f:
            f.write(error_message + f'\n\n{verified}'.upper())
    for tactic_index in local_tactic_indices(task_index, filenames):
        full_path_to_file = os.path.join(input_dir, f'task-{task_index}-{tactic_index}.lean')
        verified, error_message = client.check_file_with_nice_output(full_path_to_file, insist_on_theorem = True)
        full_path_to_error_message_file = full_path_to_file[:-5] + '.err'
        with open(full_path_to_error_message_file, "w", encoding = "utf-8") as f:
            f.write(error_message + f'\n\n{verified}'.upper())
        if verified:
            good[ind, 0] = True
            break

client.shutdown()

all_with_equivalences = np.sum(have_equivalence)
all_without_equivalences = have_equivalence.size - all_with_equivalences
good_without_equiv = np.sum(good[:,0] & np.logical_not(have_equivalence))
good_with_equiv_two_sides = np.sum(good[:,0] & good[:,1] & have_equivalence)
good_with_equiv_to_right = np.sum(good[:,0] & np.logical_not(good[:,1]) & have_equivalence)
good_with_equiv_to_left = np.sum(good[:,1] & np.logical_not(good[:,0]) & have_equivalence)

sign = {True: '+', False: '-'}
with open(os.path.join(input_dir, 'multiple_proofs_log.txt'), 'w', encoding ='utf-8') as file:
    file.write(f'Among all {all_without_equivalences} theorems without equivalences {good_without_equiv} theorems are proved\n')
    file.write(f'Among all {all_with_equivalences} theorems with equivalences {good_with_equiv_two_sides} theorems are proved to both sides\n')
    file.write(f'Among all {all_with_equivalences} theorems with equivalences {good_with_equiv_to_right} theorems are proved only to right side\n')
    file.write(f'Among all {all_with_equivalences} theorems with equivalences {good_with_equiv_to_left} theorems are proved only to left side\n')
    file.write(f'Total result: {good_without_equiv + good_with_equiv_two_sides + good_with_equiv_to_right} out of {all_without_equivalences + all_with_equivalences}\n')
    for task_index, equivalence, ver in zip(task_indices, have_equivalence, good):
        file.write(f'{task_index} {sign[ver[0]]}')
        if equivalence:
            file.write(f' {sign[ver[1]]}')
        file.write('\n')
