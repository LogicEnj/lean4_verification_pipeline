from custom_tools.batch_sender import make_batch_query
from custom_tools.model_tokenizer import get_tokenizer
import numpy as np
from sys import stdout

tokenizer = get_tokenizer("Qwen")

def get_tokenized_prompt(modified_row, tokenizer, get_text_prompt) :
    prompt = get_text_prompt(modified_row)
    messages = [
        {"role": "system", "content": "You are an expert in mathematics"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return text

def get_qwen_answer(df, batch_size, modify_row, tokenizer, get_text_prompt):
    # Process in batches of batch_size
    qwen_answer = np.empty(len(df), dtype = object)
    print(f'{len(df)} Records')
    for batch_num, batch_df in df.groupby(df.index // batch_size):
        stdout.write(f"\rProcessing batch {batch_num} ({len(batch_df)} rows)")
        stdout.flush()
        
        # Generate formatted prompts
        prompts = batch_df.apply(
            lambda row: get_tokenized_prompt(modify_row(row), tokenizer, get_text_prompt), 
            axis=1
        ).tolist()
        
        # Get LLM responses
        output_texts = make_batch_query(prompts)
        
        # Update files
        for local_index, output_text in enumerate(output_texts):
            qwen_answer[batch_num * batch_size + local_index] = output_text
    stdout.write('\n')

    return qwen_answer

def filter_answer(qwen_answer):
    def _filter_(answers):
        for text in answers:
            if '</think>' in text:
                yield text.split('</think>')[-1].strip()
            else:
                yield ''
    return list(_filter_(qwen_answer))
