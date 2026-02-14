from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.batch_sender import make_batch_query
from custom_tools.model_tokenizer import get_tokenizer
import pandas as pd
import requests
import os

def get_tokenized_prompt(row, tokenizer) :
    problem = row['problem']
    solution = row['structured_solution']
    theorem = row['theorem']

    prompt = """I give you a math problem, its structured solution, and its Lean 4 formalization. 
Please translate the stuctured solution into the proof of the theorem in a step-by-step manner so that
each step of the informal proof becomes a 'have' clause in Lean. Use 'calc' tactics for chains of calculations. 
Avoid omega, tauto and nlinarith tactics since they often lead to proofs that are not accepted by Lean 4.
Put your final Lean 4 output into the ```lean4``` code block."""
    prompt += f"Problem: \n{problem}\n"
    prompt += f"Solution: \n{solution}\n"
    prompt += f"Theorem: \n{theorem}\n"

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

######################################################################
input_file = get_input_file_or_folder('./config.yaml')
output_dir = get_output_file_or_folder('./config.yaml')


print('input:', input_file)
print('output:', output_dir)

tokenizer = get_tokenizer("Kimina-Prover")

df = pd.read_csv(input_file)

os.makedirs(output_dir, exist_ok=True)

batch_size = 10
for batch_num, batch_df in df.groupby(df.index // batch_size):
    print(f"Processing batch {batch_num} ({len(batch_df)} rows)")
    
    # Generate formatted prompts
    prompts = batch_df.apply(
        lambda row: get_tokenized_prompt(row, tokenizer), 
        axis=1
    ).tolist()
    
    # Get LLM responses
    output_texts = make_batch_query(prompts)
    
    # Update files
    for local_index, output_text in enumerate(output_texts):
        global_index = batch_num * batch_size + local_index
        
        # Extract proof from output_text
        proof = ""
        try:
            proof = output_text.split('```lean4')[-1]
            proof = proof.split('```')[0]
        except:
            print("No proof for Theorem ", global_index)

        filename = f"theorem_{global_index}.lean"

        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(proof)
