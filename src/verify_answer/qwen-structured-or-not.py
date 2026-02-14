from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
from custom_tools.batch_sender import make_batch_query
from custom_tools.model_tokenizer import get_tokenizer
import requests
import pandas as pd
import os

def get_text_prompt(row):
    problem = row['problem']
    answer = row['answer']
    
    prompt = """You are a math expert. Please solve the following problem by reasoning step by step 
and then summarize your solution in propositional logic so that it 
satisfies the following criteria: The solution should be split into several easy to understand steps, 
each written in propositional logic as A_1 ∧ ... ∧ A_n -> B as a set of premises (explicitly stated, not as A_i) and one conclusion. 
You should include all relevant premises so that the proposition is true without previous context. 
"""
    prompt += f"Problem: \n{problem}\n"
    
    return prompt

def get_tokenized_prompt(row, tokenizer) :
    prompt = get_text_prompt(row)
    messages = [
        {"role": "system", "content": "You are an expert in mathematics"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text

def get_proof(text) :
    try:
        proof = output_text.split('```lean4')[-1]
        proof = proof.split('```')[0]
        return proof
    except:
        print("No proof for a theorem")
        return ""



##################################################################
input_file = get_input_file_or_folder('./config.yaml')
output_file = get_output_file_or_folder('./config.yaml')

print('input:', input_file)
print('output:', output_file)

df = pd.read_csv(input_file)
tokenizer = get_tokenizer("Qwen")

output_dir = ('./logs')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
first_row = df.iloc[0].to_dict()
with open(os.path.join(output_dir, "prompt.txt"), 'w') as file:
    file.write(get_text_prompt(first_row))

df['structured_solution'] = None

# Process in batches of batch_size
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
        structured_solution = output_text.split('assistant')[-1]
        df.at[global_index, 'structured_solution'] = structured_solution

        with open(os.path.join(output_dir, f"solution_{global_index}.txt"), "w", encoding="utf-8") as f:
            f.write(structured_solution)

df.to_csv(output_file, index = False)


# Write to file (UTF-8 for Lean Unicode symbols)
