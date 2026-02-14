import requests
import pandas as pd
import os
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.model_tokenizer import get_tokenizer
from sys import stdout
import json

###########################################################################
input_file = get_input_file_or_folder('./config.yaml')
output_file = get_output_file_or_folder('./config.yaml')

print('input:', input_file)
print('output:', output_file)

log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, 'r', encoding = 'utf-8') as f:
    df = pd.DataFrame(json.loads(line) for line in f.readlines())

tokenizer = get_tokenizer("Kimina-Autoformalizer")

df['theorem'] = None

prompt_without_introduced = r""" Using the types of variables, please autoformalize the following problem in Lean 4. Use the following theorem names: THEOREM.
Example
What integer x satisfies 2 * x + 3 = 13?
The answer is 5
Types of variables:
$x\in\mathbb{Z}$
Formalization: theorem THEOREM (x : ℤ ) : 2 * x + 3 = 13 ↔ x = 5 := by sorry
Example
Solve \( 2 ^ x + \frac{3}{2} = \frac{19}{2} \)
The answer is x = 3
Types of variables:
$x\in\mathbb{R}$
Formalization: theorem THEOREM (x : ℝ ) : (2 : ℝ) ^ x + (3 : ℝ) / (2 : ℝ )  = (19 : ℝ) / (2 : ℝ) ↔ x = 3 := by sorry
End of Examples

"""

prompt_with_introduced = r""" Using the introduced variables, please autoformalize the following problem in Lean 4. Use the following theorem names: THEOREM.
Example
What power of 27 is equal to 81? The answer is $\frac{4}{3}$
Introduced variables:
x satisfying the equation $27^x = 81$, $x\in\mathbb{R}$
Formalization:
theorem THEOREM (x : ℝ ) : (27 : ℝ) ^ x = 81 ↔ x = (4 : ℝ) / 3 := by sorry
Example
Hillary has eleven coins, all dimes and nickels. In total, the coins are worth 75 cents. How many nickels does she have? The answer is n = 7
Introduced variables:
d is the number of dimes, $d\in\mathbb{N}$
n is the number of nickels, $n\in\mathbb{N}$
Formalization: theorem THEOREM (d n : ℕ) (h0: 10 * d + 5 * n = 75) (h1: d + n = 11) : n = 7 := by sorry
End of Examples

"""
print(f'{len(df)} Records')
for index, row in df.iterrows():

    stdout.write(f"\rRecord#{index}")
    stdout.flush()
    
    problem = row['problem']
    answer = row['answer']

    if row['was_variables'] or not row['variables']:
        prompt = prompt_without_introduced + f"\n{problem}\n The answer is {answer}\nTypes of variables:\n" + '\n'.join(row['types'])
    else:
        prompt = prompt_with_introduced + f"\n{problem}\n The answer is {answer}. Introduced variables: \n" + '\n'.join(f"{definition}, {typ}" for definition, typ in zip(row['definitions'], row['types'])) 

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
    formalization = output_text.split('assistant')[-1]

    df.at[index, 'theorem'] = formalization

    filename = f"theorem_{index}.lean"

    with open(os.path.join(log_dir, filename), "w", encoding="utf-8") as f:
        f.write(formalization)

stdout.write('\n')

df = df.drop(columns = ['was_variables', 'variables', 'types', 'definitions', 'constants', 'full_descriptions'])
df.to_csv(output_file, index = False)
