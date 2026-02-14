import requests
import pandas as pd
import os
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
from custom_tools.model_tokenizer import get_tokenizer

###########################################################################
input_file = get_input_file_or_folder('./config.yaml')
output_file = get_output_file_or_folder('./config.yaml')

print('input:', input_file)
print('output:', output_file)

log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

df = pd.read_json(input_file, lines=True)

tokenizer = get_tokenizer("Kimina-Autoformalizer")

df['theorem'] = None

for index, row in df.iterrows():
    
    problem = row['problem']
    answer = row['answer']

    prompt = """ I give you a math problem and its answer. Please formalize the statement that the provided answer
is indeed the answer to the problem as a theorem in Lean 4, as in the following three examples: 
Example
Problem: what integer x satisfies 2 * x + 3 = 13 
Answer: x = 5
Formalization: theorem t (x : ℤ ) : 2 * x + 3 = 13 ↔ x = 5
Example
Problem: What power of 4 is equal to half of 16? Express your answer as a common fraction.
Answer: \\frac{3}{2}
Formalization: theorem t (x : ℝ ) : (4 : ℝ) ^ x = (16 : ℝ) / (2 : ℝ) ↔ x = (3 : ℝ) / 2 
Example
Problem: solve \( 2 ^ x + \\frac{3}{2} = \\frac{19}{2} \)
Answer: x = 3
Formalization: theorem t (x : ℝ ) : (2 : ℝ) ^ x + (3 : ℝ) / (2 : ℝ )  = (19 : ℝ) / (2 : ℝ) ↔ x = 3 
End of Examples\n\n"""


    prompt += f"Problem: \n{problem}\n"
    prompt += f"Answer: \n{answer}\n"

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

df.to_csv(output_file, index = False)
