from correct_funcs import correct_func, find_symb_outside_braces
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
import pandas as pd
import json
import re
from sys import stdin

def correct(theorems, lemmas):
    lemma_lookup = re.compile(r'theorem.*:=', flags = re.M | re.DOTALL)
    theorem_lookup = re.compile(r'theorem[^{(:]*')
    for j, theorem in enumerate(theorems):
        #match = lemma_lookup.search(theorem)
        #if match:
        #    theorem_match = theorem_lookup.search(theorem)
        #    begin_theorem_stat = theorem_match.end()
        #    end_theorem_stat = match[0].find(':=') + theorem_match.start()
        #    theorem_stat = theorem[begin_theorem_stat: end_theorem_stat]
        #    pos = find_symb_outside_braces(':', theorem_stat)
        #    if pos >= 0:
        #        dict_for_formulas = {}
        #        for dct in [json.loads(row['formula_dict']) for _, row in lemmas.iterrows() if row['task_number'] == j]:
        #            dict_for_formulas.update(dct)
        #        new_hyps = ' '.join(f'(H_{var} : {hyp})' for var, hyp in dict_for_formulas.items() if var not in theorem_stat)
        #        if new_hyps:
        #            theorem_stat = theorem_stat[: pos] + ' ' + new_hyps + theorem_stat[pos :]
        #            theorem = theorem[: begin_theorem_stat] + theorem_stat + theorem[end_theorem_stat :]
        theorem = theorem.replace('\r\n', '\n')
        unexpected_import = re.search(r'theorem.*import', theorem, flags = re.M|re.DOTALL)
        if unexpected_import:
            theorem = theorem[:unexpected_import.start()] + theorem[theorem.rfind('theorem'):]
        theorem = correct_func(theorem).replace('<->','↔').replace('->','→')
        yield theorem

input_lemma = get_input_file_or_folder('./config.yaml', 1)
input_file = get_input_file_or_folder('./config.yaml', 0)
output_file = get_output_file_or_folder('./config.yaml')

print('input:', input_lemma, input_file)
print('output:', output_file)

df = pd.read_csv(input_file)
lemmas = pd.read_csv(input_lemma)
lemmas = lemmas[['task_number', 'formula_dict']]
df['theorem'] = list(correct(df['theorem'], lemmas))
df['correct'] = True
df.to_csv(output_file, index = False)
