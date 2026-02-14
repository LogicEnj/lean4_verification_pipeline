from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.batch_sender import make_batch_query
from custom_tools.model_tokenizer import get_tokenizer
import pandas as pd
import requests
import re
import os
from sys import stdout
from correct_funcs import correct_func
from custom_tools.lean_client import LSPClient

def get_tokenized_prompt(formal_lemma_text, tokenizer) :
    
    prompt = """I give you a very easy theorem statement formalized in Lean 4. Do not change it.
Please supply the proof of this theorem. Use 'calc' tactics for chains of calculations. 
Avoid omega, tauto and nlinarith tactics since they often lead to proofs that are not accepted by Lean 4.
Put your final Lean 4 output into the ```lean4``` code block.
"""
    prompt += f"Theorem: \n{formal_lemma_text}\n"

    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def insert_proof(statement):
    for line in statement.split('\n'):
        if 'sorry' in line:
            indent = '  ' + re.search(r'^(\s*)',line)[1]
            yield line.replace('sorry', f'\n{indent}try exact?\n{indent}try norm_num\n{indent}try linarith\n{indent}try simp_all')
        else:
            yield line

def check_proof(proof, client, file_uri):
    if proof:
        verified, error_message = client.check_text(proof, file_uri)
        if 'sorry' in proof:
            verified = False
            error_message = 'sorry in proof'
    else:
        verified = False
        error_message = 'empty proof'
    return verified, error_message
    

#####################################################################################

tokenizer = get_tokenizer("Kimina-Prover")

input_dir = get_input_file_or_folder('./config.yaml')
output_dir = get_output_file_or_folder('./config.yaml')
os.makedirs(output_dir, exist_ok=True)

print('input_dir:', input_dir)
print('output_dir:', output_dir)

fact_pattern = re.compile(r'^fact_\d+_\d+_\d+\.lean$')
lemma_pattern = re.compile(r'^thm_\d+_\d+\.lean$')
df = []

names = os.listdir(input_dir)
print(f'{len(names)} input files')
client = LSPClient()
for i, filename in enumerate(names):
    stdout.write(f'\r input file #{i}')
    stdout.flush()
    full_path_to_file = os.path.join(input_dir, filename)
    if os.path.isfile(full_path_to_file) and (fact_pattern.fullmatch(filename) or lemma_pattern.fullmatch(filename)):
        with open(full_path_to_file, 'r') as file:
            formal_lemma_text = file.read()
        if 'sorry' in formal_lemma_text:
            proof = '\n'.join(insert_proof(formal_lemma_text))
        else:
            proof = formal_lemma_text
        verified, error_message = check_proof(proof, client, filename + 'manual')
        if verified:
            full_path_to_output_file = os.path.join(output_dir, filename)
            full_path_to_err = full_path_to_output_file[:-5] + '.err'
            with open(full_path_to_output_file, 'w', encoding = 'utf-8') as file:
                file.write(proof)
            with open(full_path_to_err, 'w', encoding = 'utf-8') as file:
                file.write(error_message + '\n\nTRUE')
        else:
            df.append([filename, formal_lemma_text])
stdout.write('\n')

df = pd.DataFrame(df, columns = ['filename', 'formal_lemma'])

print(f'{len(df)} Records')
batch_size = 21
for batch_num, batch_df in df.groupby(df.index // batch_size):
    stdout.write(f"\rProcessing batch {batch_num} ({len(batch_df)} rows)")
    stdout.flush()
    
    # Get LLM responses
    output_texts = make_batch_query([get_tokenized_prompt(formal_lemma, tokenizer) for formal_lemma in batch_df['formal_lemma']])

    # Update files
    for (_, row), output_text in zip(batch_df.iterrows(), output_texts):
        match = re.search(r'```lean4.*```', output_text, flags = re.M|re.DOTALL)
        if match:
            proof = match[0]
            proof = proof[proof.rfind('```lean4') + 8 : ]
            proof = proof[: proof.find('```')]
            proof = correct_func(proof).replace(' in ', ' ∈ ').replace('sqrt_eq_iff_sq_eq', 'sqrt_eq_iff_eq_sq').replace('div_lt_div_right', 'div_lt_div_iff_of_pos_right').strip()
        if not proof:
            proof = row['formal_lemma']
        full_path_to_output_file = os.path.join(output_dir, row['filename'])
        full_path_to_err = full_path_to_output_file[:-5] + '.err'
        with open(os.path.join(output_dir, row['filename']), "w", encoding="utf-8") as file:
            file.write(proof)
stdout.write('\n')
for filename in df['filename']:
    full_path_to_output_file = os.path.join(output_dir, filename)
    full_path_to_err = full_path_to_output_file[:-5] + '.err'
    with open(full_path_to_output_file, "r", encoding="utf-8") as file:
        verified, error_message = check_proof(file.read(), client, filename)
    with open(full_path_to_err, "w", encoding="utf-8") as file:
        file.write(error_message + f'\n\n{verified}'.upper())

client.shutdown()
os.system('cp ' + os.path.join(input_dir, 'theorems.csv') + ' ' + os.path.join(output_dir, 'theorems.csv'))
