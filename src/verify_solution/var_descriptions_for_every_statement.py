from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
import pandas as pd
import os
import re
import json
from get_vars_from_statement import get_descriptions

##################################################################
problems_json = get_input_file_or_folder('./config.yaml', 0)
lemmas_csv = get_input_file_or_folder('./config.yaml', 1)
output_dir = get_output_file_or_folder('./config.yaml')
os.makedirs(output_dir, exist_ok=True)

print('input:', problems_json, lemmas_csv)
print('output:', output_dir)

batch_size = 10
lemmas = pd.read_csv(lemmas_csv)
lemmas = lemmas[lemmas["included"]==True]

_, vars_types_consts_fulls = get_descriptions(lemmas['lemma'], 'statement', batch_size)

for (_, row), element in zip(lemmas.iterrows(), vars_types_consts_fulls):
    filename = f'thm_{row["task_number"]}_{row["lemma_number"]}.json'
    dct = {'lemma': row['lemma']}
    dct['variables'], dct['types'], dct['constants'], dct['full_descriptions'], dct['raw_output'] = element
    with open(os.path.join(output_dir, filename), 'w', encoding = 'utf-8') as file:
        file.write(json.dumps(dct))

keys = ['problem', 'variables', 'types', 'constants', 'full_descriptions', 'definitions', 'raw_output']
with open(problems_json, 'r', encoding = 'utf-8') as file:
    lines = file.readlines()
for i, line in enumerate(lines):
    line_parsed = json.loads(line)
    dct = dict((key, line_parsed[key]) for key in keys)
    with open(os.path.join(output_dir, f'thm_{i}.json'), 'w', encoding = 'utf-8') as file:
        file.write(json.dumps(dct))
