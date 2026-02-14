from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.model_tokenizer import get_tokenizer
from custom_tools.lean_client import LSPClient
import requests
import pandas as pd
import re
import os
import json
from sys import stdin

def get_fact_indices(directory):
    fact_pattern = re.compile(r'^fact_(\d+)_(\d+)_(\d+)\.lean$')
    def _iter_fact_indices(directory):
        for name in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, name)):
                match = fact_pattern.fullmatch(name)
                if match:
                    yield (int(match[1]), int(match[2]), int(match[3]))
    return sorted(list(_iter_fact_indices(directory)))

def get_lemma_indices(directory):
    lemma_pattern = re.compile(r'^thm_(\d+)_(\d+)\.lean$')
    def _iter_lemma_indices(directory):
        for name in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, name)):
                match = lemma_pattern.fullmatch(name)
                if match:
                    yield (int(match[1]), int(match[2]))
    return sorted(list(_iter_lemma_indices(directory)))

tokenizer = get_tokenizer("Kimina-Prover")
prompt_start = """I give you a very easy theorem statement formalized in Lean 4. Do not change it.
Please supply the proof of this theorem. Use 'calc' tactics for chains of calculations.
Avoid omega, tauto and nlinarith tactics since they often lead to proofs that are not accepted by Lean 4.
Put your final Lean 4 output into the ```lean4``` code block.
"""
def get_proof(formal_text):
    prompt = prompt_start + f"Theorem: \n{formal_text}\n"

    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
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
    output_text = output_text.split('assistant')[-1]

    # Extract proof from output_text
    match = re.search(r'```lean4.*```', output_text, flags = re.M|re.DOTALL)
    if not match:
        proof = ''
    else:
        proof = match[0]
        proof = proof[proof.rfind('```lean4') + 8 : ] #remove ```lean4
        proof = proof[: proof.find('```')].strip()
        proof = re.sub(r"\\n'?,?\s*'?", '\n', proof, flags = re.M)
        proof = re.sub(r'/-.*-/', '', proof, flags = re.M | re.DOTALL) ##remove comment (can contains sorry)
        proof = re.sub(r'\n *--.*','', proof, flags = re.M) ##1:remove comments starting with --
        match = re.search(r'^ *· --.*\n.*', proof, flags = re.M)
        while match:
            proof = re.sub(r'^( *)· --.*\n *(.*)', r'\1· \2', proof, flags = re.M) ##2:remove comments starting with --
            match = re.search(r'^ *· --.*\n.*', proof, flags = re.M)
        proof = proof.replace(' in ', ' ∈ ').replace('sqrt_eq_iff_sq_eq', 'sqrt_eq_iff_eq_sq').replace('div_lt_div_right', 'div_lt_div_iff_of_pos_right')

    return proof.strip()

def verification(statement, formalization, correct_lemmas, client, filename):
    def check_proof(proof, client, file_uri):
        if proof:
            checked, error_message = client.check_text(proof, file_uri)
            if 'sorry' in proof:
                checked = False
                error_message = 'sorry in proof'
        else:
            checked = False
            error_message = 'Empty Proof'
        return checked, error_message

    def insert_proof(statement):
        for line in statement.split('\n'):
            if 'sorry' in line:
                indent = '  ' + re.search(r'^(\s*)',line)[1]
                yield line.replace('sorry', f'\n{indent}try exact?\n{indent}try norm_num\n{indent}try linarith\n{indent}try simp_all')
            else:
                yield line

    info = '##########Info\n' + formalization + '\n\n' + '\n\n'.join(correct_lemmas) + '\n\n'
    answer = 'e'
    edit = False
    while answer=='e':
        print(f'##########Statement\n{statement}\n##########End of statement\n')
        if 'sorry' in statement:
            proof = '\n'.join(insert_proof(statement))
        else:
            proof = statement
        checked, error_message = check_proof(proof, client, filename + 'manual')
        if checked:
            error_message += '\n\nTRUE'
            answer = 'y'
        else:
            proof = get_proof(statement)
            checked, error_message = check_proof(proof, client, filename)
            if checked:
                error_message += '\n\nTRUE'
                answer = 'y'
            else:
                print(f'{info}{statement}\n##########End info\n\n')
                answer = ''
                while answer not in ('y', 'n', 'e', 'r', 'c'):
                    answer = input('''This theorem was failed to be proven. Write:
"y" (yes), if you believe it to be true
"e" (edit), if there were mistakes with formalization
"n" (no), if you believe it to be false
"r" (retry), if you want a new proof
"c" (continue), if you want continue without including the statement
'''
                    )
                if answer == 'e':
                    print('Enter new statement, press Enter and then Ctrl+D')
                    statement = stdin.read()
                    edit = True
                elif answer == 'r':
                    answer = 'e'
                else:
                    error_message += '\n\nFALSE'
                    proof = statement
    match answer:
        case 'c':
            ret_val = -1
        case 'n':
            ret_val = 0
        case 'y':
            ret_val = int(edit) + 1
    return proof, ret_val, statement, error_message

input_dir = get_input_file_or_folder('./config.yaml')
output_dir = get_output_file_or_folder('./config.yaml')
os.makedirs(output_dir, exist_ok=True)

print('input:', input_dir)
print('output_dir:', output_dir)

problems = pd.read_csv(os.path.join(input_dir, 'theorems.csv'))
problems['checked'] = None
lemma_indices = get_lemma_indices(input_dir)
fact_indices = get_fact_indices(input_dir)

ret_val_dct = {-1: "Skipped", 0: "Refused", 1: "Accepted", 2: "Edited"}
correct_lemmas = []
client = LSPClient()
for task_number, row in problems.iterrows():
    if not row['correct']:
        continue
    formalization = row['direct']

    check_log = ''
    all_checked = True
    correct_lemmas.clear()
    lemma_numbers = [n_lemma for n_task, n_lemma in lemma_indices if n_task == task_number]
    for lemma_number in lemma_numbers:
        fact_numbers = [n_fact for n_task, n_lemma, n_fact in fact_indices if n_task == task_number and n_lemma == lemma_number]
        for fact_number in fact_numbers:
            print(f'Lemma {lemma_number}, Fact {fact_number}')
            fact_name = f'fact_{task_number}_{lemma_number}_{fact_number}.lean'
            input_fact_path = os.path.join(input_dir, fact_name)
            output_fact_path = os.path.join(output_dir, fact_name)
            with open(input_fact_path, 'r', encoding = 'utf-8') as f:
                fact = f.read()
            proof, ret_val, fact, error_message = verification(fact, formalization, correct_lemmas, client, fact_name)
            check_log += fact_name[:-5] + ': ' + ret_val_dct[ret_val] + '\n'
            if ret_val > 0:
                with open(output_fact_path, 'w', encoding = 'utf-8') as f:
                    f.write(proof)
                with open(output_fact_path[:-4] + 'err', 'w', encoding = 'utf-8') as f:
                    f.write(error_message)
                correct_lemmas.append(fact)
            elif ret_val == 0:
                all_checked = False
                check_log += 'FALSE'
                break
        if not all_checked:
            break
        print(f'Lemma {lemma_number}')
        lemma_name = f'thm_{task_number}_{lemma_number}.lean'
        input_lemma_path = os.path.join(input_dir, lemma_name)
        output_lemma_path = os.path.join(output_dir, lemma_name)
        with open(input_lemma_path, 'r', encoding = 'utf-8') as f:
            lemma = f.read()
        proof, ret_val, lemma, error_message = verification(lemma, formalization, correct_lemmas, client, lemma_name)
        check_log += lemma_name[:-5] + ': ' + ret_val_dct[ret_val] + '\n'
        if ret_val > 0:
            with open(output_lemma_path, 'w', encoding = 'utf-8') as f:
                f.write(proof)
            with open(output_lemma_path[:-4] + 'err', 'w', encoding = 'utf-8') as f:
                f.write(error_message)
            correct_lemmas.append(lemma)
        elif ret_val == 0:
            all_checked = False
            check_log += 'FALSE'
            break

    if all_checked:
        check_log += 'TRUE'
    problems.at[task_number, 'checked'] = check_log
client.shutdown()
problems.to_csv(os.path.join(output_dir, 'theorems.csv'), index = False)
