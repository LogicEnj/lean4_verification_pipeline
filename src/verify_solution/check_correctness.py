from custom_tools.config_reader import get_input_file_or_folder
import pandas as pd
import os

input_dir = get_input_file_or_folder('./config.yaml')
theorems = pd.read_csv(os.path.join(input_dir, 'theorems.csv'))

any_correct = any(theorems['correct'])
if any_correct:
    print('there_are_correct_formalizations')
else:
    print('all_formalizations_are_incorrect')
