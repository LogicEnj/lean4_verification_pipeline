from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
import os
import json
import re

def problem_indices(names):
    problem_pattern = re.compile(r'^thm_(\d+)\.json$')
    for name in names:
        match = problem_pattern.fullmatch(name)
        if match:
            yield int(match[1])

def lemma_indices(names, ind):
    lemma_pattern = re.compile(rf'^thm_{ind}_(\d+)\.json$')
    for name in names:
        match = lemma_pattern.fullmatch(name)
        if match:
            yield int(match[1])

##################################################################
input_dir = get_input_file_or_folder('./config.yaml')
output_dir = get_output_file_or_folder('./config.yaml')
os.makedirs(output_dir, exist_ok=True)

print('input:', input_dir)
print('output:', output_dir)

names = [name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))]

for problem_file in list(problem_indices(names)):
    with open(os.path.join(input_dir, f'thm_{problem_file}.json'), 'r', encoding = 'utf-8') as f:
        info = json.loads(f.read())
    try:
        del info['full_descriptions']
    except:
        pass
    with open(os.path.join(output_dir, f'thm_{problem_file}.json'), 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(info))
    all_variables = info['variables']
    all_types = info['types']
    all_constants = info['constants']
    for lemma_file in sorted(list(lemma_indices(names, problem_file))):
        with open(os.path.join(input_dir, f'thm_{problem_file}_{lemma_file}.json'), 'r', encoding = 'utf-8') as f:
            info = json.loads(f.read())
        try:
            del info['full_descriptions']
        except:
            pass
        for i, (var, typ, const) in enumerate(zip(info['variables'], info['types'], info['constants'])):
            try:
                ind_in_all = all_variables.index(var)
                info['types'][i] = all_types[ind_in_all]
                info['constants'][i] = all_constants[ind_in_all]
            except ValueError:
                all_variables.append(var)
                all_types.append(typ)
                all_constants.append(const)
                info['constants'][i] = const
        with open(os.path.join(output_dir, f'thm_{problem_file}_{lemma_file}.json'), 'w', encoding = 'utf-8') as f:
            f.write(json.dumps(info))
