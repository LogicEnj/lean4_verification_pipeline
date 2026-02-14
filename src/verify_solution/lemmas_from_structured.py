from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.model_tokenizer import get_tokenizer
import requests
import json
import os
import re
import pandas as pd
from sys import stdout

def sel_correct_names(input_dir):
    lemma_pattern = re.compile(r'^thm_(\d+)_(\d+)\.json$')
    for name in os.listdir(input_dir):
        match = lemma_pattern.fullmatch(name)
        if match and os.path.isfile(os.path.join(input_dir, name)):
            yield (int(match[1]), int(match[2])) #match[1] is task_number, match[2] is lemma_number

def get_prompts(line):
    def _get_first_part_(substring):
        if not r'\land' in substring:
            return f'Given that {substring}, '
        else:
            return 'Given a set of premises: ' + ' and '.join(hyp.strip() for hyp in substring.split(r'\land')) + ', '

    if not '->' in line:
        return f'Prove that {line}.'
    else:
        statement, right = (part.strip() for part in line.split('->', 1))
        result = _get_first_part_(statement)
        if right:
            result = result + f'prove that {right}.'
        return result

def create_var_dict(var_list, var_defs, statement, lemma):
    accepted = '0123456789-+/*^ _().'
    def is_seminum(line):
        line = line.replace(r'\log', '').replace(r'\sqrt', '').replace(r'\frac', '').replace('{', '').replace('}', '')
        return all(c in accepted for c in line)

    def leanify(hyp):
        def extract_arg(part):
            if part.startswith('{'):
                balance = 1
                for i in range(1, len(part)):
                    match part[i]:
                        case '{':
                            balance += 1
                        case '}':
                            balance -= 1
                            if balance == 0:
                                break
                    i += 1
                num = part[1: i]
            else:
                i = 0
                num = part[0]
            return part[i+1:], num

        def two_args(part):
            part, first_num = extract_arg(part)
            part, second_num = extract_arg(part)
            return first_num, second_num, part

        hyp = hyp.replace(r'\sqrt', ' Real.sqrt ')
        log_pos = hyp.find(r'\log_')
        while log_pos > -1:
            first_num, second_num, remaining = two_args(hyp[log_pos+5 : ])
            hyp = hyp[: log_pos] + f' Real.logb ({first_num}) ({second_num})' + remaining
            log_pos = hyp.find(r'\log_')
        frac_pos = hyp.find(r'\frac')
        while frac_pos > -1:
            first_num, second_num, remaining = two_args(hyp[frac_pos+5 : ])
            hyp = hyp[: frac_pos] + f'({first_num}) / ({second_num})' + remaining
            frac_pos = hyp.find(r'\frac')
        return hyp.replace('{', '(').replace('}', ')').strip()

    var_dict = {}
    for hyp in lemma.split('->', 1)[0].split(r'\land'):
        hyp = hyp.strip()
        if hyp in statement:
            continue
        hyp_without_dollars = hyp.replace('$','').strip()
        if hyp_without_dollars.count('=') == 1:
            r, l = (part.strip() for part in hyp_without_dollars.split('='))
            if r in var_list and is_seminum(l):
                var = r
                const = l
            elif l in var_list and is_seminum(r):
                var = l
                const = r
            else:
                continue
            if not var_defs:
                continue
            var_index = var_list.index(var)
            if const in var_defs[var_index]:
                value = leanify(hyp_without_dollars)
                if value:
                    var_dict[var] = value
            var_list.pop(var_index)
            var_defs.pop(var_index)
    return var_dict
    
tokenizer = get_tokenizer("Kimina-Autoformalizer")

input_dir = get_input_file_or_folder('./config.yaml')
output_csv = get_output_file_or_folder('./config.yaml')

print('input:', input_dir)
print('output:', output_csv)

indices = sorted(list(sel_correct_names(input_dir)))

print(len(indices), "records")

df = []
for i, (task_number, lemma_number) in enumerate(indices):
    stdout.write(f'\rRecord#{i}')
    stdout.flush()
    with open(os.path.join(input_dir, f'thm_{task_number}_{lemma_number}.json'), 'r', encoding = 'utf-8') as f:
        lemma_info = json.loads(f.read())
    prompt = 'Please autoformalize the following problem in Lean 4 as given without modifying equalities. Use the following theorem names: THEOREM. \n' + get_prompts(lemma_info["lemma"])
    vars_info = '\n'.join(typ for typ in lemma_info['types'])
    if vars_info:
        system_prompt = "\nTypes of the variables are the following:\n" + vars_info
        try:
            ind_of_i = lemma_info['variables'].index('i')
            if lemma_info['constants'][ind_of_i]:
                system_prompt += '\nUse the denotation Complex.I for the complex unity i'
        except ValueError:
            pass
    else:
        system_prompt = ''
    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4." + system_prompt},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Query the vLLM server
    API_URL = "http://localhost:8000/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": text,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096
    }

    response = requests.post(API_URL, headers=headers, json=data, proxies={"http": None, "https": None}, verify=False)
    output_text = response.json()["text"][0]
    formalization = output_text.split('assistant')[-1]

    with open(os.path.join(input_dir, f'thm_{task_number}.json'), 'r', encoding = 'utf-8') as f:
        problem_info = json.loads(f.read())
    var_dict = create_var_dict(problem_info['variables'], problem_info['definitions'], problem_info['problem'], lemma_info["lemma"])
    df.append({"task_number" : task_number, "lemma_number" : lemma_number, "lemma" : lemma_info["lemma"], "formalization" : formalization, "formula_dict" : json.dumps(var_dict)})
if df:
    pd.DataFrame(df).to_csv(output_csv, index = False)
else:
    with open(output_csv, 'w', encoding = 'utf-8') as f:
        f.write('task_number,lemma_number,lemma,formalization,formula_dict')
stdout.write('\n')
