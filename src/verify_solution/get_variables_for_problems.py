from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
import os
import re
import json
import pandas as pd
from get_vars_from_statement import get_descriptions, tokenizer, get_qwen_answer, filter_answer
import argparse

parser = argparse.ArgumentParser(description='Introduce variables')
parser.add_argument('--introduce', type=int, required=True, help='introduce variables')
introduce = bool(parser.parse_args().introduce)
print(f"Introduce variables: {introduce}")

def get_text_prompt_without_variables(row):
    prompt = "You are a math expert. For the following problem, introduce variables, their types and definitions to solve the problem, but do not solve the problem. Name of each variable must be a latin letter that can contain a numerical index. Separate each variable, each type and each definition by new line and enumerate them."
    prompt += r"""
I will give you examples of variables, their types and definitions:

begin of example section
Problem:
Compute the product of the number $1-sqrt{2}$ and its radical conjugate.
Variables:
1. a
2. b
Types:
1. a\in\mathbb{R}
2. b\in\mathbb{R}
Definitions:
1. a is $1-\sqrt{2}$
2. b is $1+\sqrt{2}$

Problem:
Siddhartha Gautama has 17 coins, all uranium and plutonium. Uranium and plutonium coin costs 4 and 7 rupees, respectively. In total, the coins are worth 74 rupees. How many uranium coins does he have?
Variables:
1. u
2. p
Types:
1. u\in\mathbb{N}
2. p\in\mathbb{N}
Definitions:
1. u is the number of uranium coins
2. p is the number of plutonium coins
end of example section

Put variables between
### Variables start
and
### Variables end
headers.

Put types between
### Types start
and
### Types end
headers.

Put definitions between
### Definitions start
and
### Definitions end
headers.

"""
    return prompt + f"Problem: \n{row['problem']}\n"

def extract_by_pattern(descriptions, pattern):
    item_pattern = re.compile('^\d+\. (.*)$')
    def common_extract(text):
        for part in text.split('\n'):
            match = item_pattern.search(part)
            if match:
                yield match[1].strip()
            else:
                yield part.strip()

    for description in descriptions:
        extracted_text = re.search(pattern, description, flags = re.M|re.DOTALL)
        if extracted_text:
            extracted_list = list(common_extract(extracted_text[1]))
            yield extracted_list
        else:
            yield []

def set_underline(variables, types, definitions):
    for i, (vars_of_problem, types_of_problem, defins_of_problem) in enumerate(zip(variables, types, definitions)):
        for j, (var, typ, defin) in enumerate(zip(vars_of_problem, types_of_problem, defins_of_problem)):
            if len(var)>1 and var[1] != '_':
                var_new = var[0] + '_' + var[1:]
                esc_var = re.escape(var)
                definitions[i][j] = re.sub(rf'\b{esc_var}\b', var_new, defin, flags = re.M)
                types[i][j] = re.sub(rf'\b{esc_var}(\b|\\in)', var_new + r'\1', typ, flags = re.M)
                variables[i][j] = var_new

def Q_to_R(types_list, problem):
    problem = problem.lower()
    if 'numerator' in problem or 'denominator' in problem:
        return types_list
    else:
        return [typ.replace(r'\mathbb{Q}', r'\mathbb{R}').replace('ℚ', 'ℝ') for typ in types_list]
##################################################################
input_file = get_input_file_or_folder('./config.yaml')
output_file = get_output_file_or_folder('./config.yaml')
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print('input:', input_file)
print('output:', output_file)

batch_size = 10

with open(input_file, 'r', encoding = 'utf-8') as file:
    data = [json.loads(line) for line in file.readlines() if line.strip()]

problems = [item['problem'] for item in data]

all_variables, vars_types_consts_fulls = get_descriptions(problems, 'problem', batch_size)

problems_without_vars = pd.DataFrame([problem for problem, problem_variables in zip(problems, all_variables) if not problem_variables], columns = ['problem'])

if introduce:
    raw_descriptions = get_qwen_answer(problems_without_vars, batch_size, lambda row: row, tokenizer, get_text_prompt_without_variables)
else:
    raw_descriptions = ['']*len(problems_without_vars)
descriptions = filter_answer(raw_descriptions)
variables = list(extract_by_pattern(descriptions, r'### Variables start\n(.*)\n### Variables end'))
types = list(extract_by_pattern(descriptions, r'### Types start\n(.*)\n### Types end'))
definitions = list(extract_by_pattern(descriptions, r'### Definitions start\n(.*)\n### Definitions end'))
set_underline(variables, types, definitions)
ind_without_vars = 0
with open(output_file, 'w', encoding = 'utf-8') as file:
    for item, problem_variables, var_type_const_full in zip(data, all_variables, vars_types_consts_fulls):
        if problem_variables:
            item['variables'], tmp_types, item['constants'], item['full_descriptions'], item['raw_output'] = var_type_const_full
            item['definitions'] = []
            item['was_variables'] = True
        else:
            item['variables'] = variables[ind_without_vars]
            tmp_types = types[ind_without_vars]
            item['definitions'] = definitions[ind_without_vars]
            item['full_descriptions'] = descriptions[ind_without_vars]
            item['raw_output'] = raw_descriptions[ind_without_vars]
            ind_without_vars += 1
            item['constants'] = [False]*len(item['variables'])
            item['was_variables'] = False
        item['types'] = list(Q_to_R(tmp_types, item['problem']))
        file.write(json.dumps(item) + '\n')
