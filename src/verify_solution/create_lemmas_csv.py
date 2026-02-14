from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
import pandas as pd
import json
import re
from sys import stdout
from correct_funcs import remove_outer_parentheses

def check_nary_system(left, right): #if a_b = c_d
    left = left.replace('{', '').replace('}', '').strip()
    right = right.replace('{', '').replace('}', '').strip()
    a, b = (part.strip() for part in left.split('_', 1))
    c, d = (part.strip() for part in right.split('_', 1))
    if a.isdigit() and c.isdigit():
        try:
            return int(a, int(b)) == int(c, int(d))
        except (TypeError, ValueError):
            return False
    else:
        return False

words = ['is', 'by', 'let', 'have', 'know', 'to', 'for', 'of', 'in']
not_pointless_seq = ['=', '|', '>', '<', r'\equiv', r'\text', r'\neq', r'\vdots', r'\propto', r'\in', r'\le', r'\ge', ':']
def not_pointless_hypotheses(hyp_list, prev_conclusion):
    concl_without_dollars = prev_conclusion.replace('$','').strip()
    if '=' in concl_without_dollars:
        concl_eq = [concl_without_dollars.split('=')[0].strip(), concl_without_dollars.split('=')[1].strip()]
    else:
        concl_eq = []
    for hyp in hyp_list:
        hyp_without_dollars = hyp.replace('$','').strip()
        if not hyp_without_dollars:
            continue
        if hyp_without_dollars.strip() in concl_eq:
            hyp = prev_conclusion
            hyp_without_dollars = concl_without_dollars
        if hyp_without_dollars.count('=') == 1:
            left, right = (part.strip() for part in hyp_without_dollars.split('='))
            if left == right or ('_' in left and '_' in right and check_nary_system(left, right)):
                continue
        if any(sym in hyp_without_dollars for sym in not_pointless_seq) or any(word in words for word in re.findall('[a-z]+', hyp_without_dollars)):
            yield hyp

def break_chain(result, sym):
    def _break_iter(result, sym):
        if sym == '=':
            var = 'parts[0]'
        else:
            var = 'parts[i-1]'
        for div_part in re.split(r'(->|\\land|\\lor|\\rightarrow|\$)', result):
            if div_part.count(sym) > 1:
                parts = [part.strip() for part in div_part.split(sym)]
                yield '(' + r' \land '.join(f'{eval(var)} {sym} {parts[i]}' for i in range(1, len(parts))) + ')'
            else:
                yield div_part.strip()
    return ' '.join(_break_iter(result, sym))

def check_hypotheses(df):
    land = re.compile(r'(\$\s*)?\\land(\s*\$)?')
    transitive_rels = ('=', '<', '>', r'\le', r'\ge')
    prev_conclusion = ''
    for _, row in df.iterrows():
        lemma = row['init_lemma'].replace(' and ', r' \land ').replace('!=', r'\neq').replace('≠', r'\neq').replace('⋮', r'\vdots').replace('≤', r'\le').replace('$$', '$').replace('≥', r'\ge').replace(r'\(',' $').replace(r'\)','$ ').replace(r'\[',' $').replace(r'\]','$ ').replace(r'\begin{align*}', '$').replace(r'\end{align*}', '$').replace('→', r'\rightarrow').replace('∨', r'\lor').replace('∧', r'\land').replace(r'\leq', r'\le').replace(r'\geq', r'\ge')
        lemma = re.sub(r'\\[a-z]frac', r'\\frac', lemma)
        lemma = re.sub(r'\\left\s*\(', '(', lemma)
        lemma = re.sub(r'\\right\s*\)', ')', lemma)
        lemma = re.sub(r'\s+', ' ', lemma)
        if not row['included']:
            yield lemma, False
            continue
        parts = [part.strip() for part in lemma.split('->', 1)]
        hyp_list = [hyp.strip() for hyp in parts[0].split(r'\land')]

        result = r' \land '.join(not_pointless_hypotheses(hyp_list, prev_conclusion))
        if '->' in lemma:
            if ''.join(not_pointless_hypotheses([parts[1]], '')).strip():
                prev_conclusion = parts[1]
                if result:
                    result += ' -> ' + prev_conclusion
                else:
                    result = prev_conclusion
            else:
                yield lemma, False
                continue
        for sym in transitive_rels: #break (in)equality chains
            result = break_chain(result, sym)
        if not result:
            yield lemma, False
            continue

        yield remove_outer_parentheses(land.sub(r'\\land', result)), True

input_file = get_input_file_or_folder('./config.yaml')
output_file = get_output_file_or_folder('./config.yaml')

print('input:', input_file)
print('output:', output_file)

with open(input_file, 'r', encoding = 'utf-8') as f:
    solutions = [json.loads(line)['structured_solution'] for line in f.readlines()]

print(len(solutions), "records")

df = []
lemma_pattern = re.compile(r'^\d+\.(.+)$')
for index, solution in enumerate(solutions):
    stdout.write(f'\rTask#{index}')
    stdout.flush()
    match = re.search(r'### Summary in Propositional Logic\n(.*)', solution, flags = re.M | re.DOTALL)
    if not match:
        continue
    solution = match[1]
    match = re.search(r'(.*)\n### End of Summary', solution, flags = re.M | re.DOTALL)
    end = False
    if match:
        solution = match[1]
        end = True
    structured_solution = solution.split('\n')
    if end:
        last_lemma = ''
    else:
        last_lemma = structured_solution[-1].strip()
        structured_solution = structured_solution[:-1]
    lemma_old = ''
    lemma_number = 1
    for line in structured_solution:
        match = lemma_pattern.fullmatch(line.strip())
        if match:
            full_lemma = match[1].strip()
            if not full_lemma:
                continue
            included = full_lemma != lemma_old
            if full_lemma.count('->') > 1:
                parts = full_lemma.split('->')
                for k in range(len(parts) - 1):
                    dct = {"task_number": index, "lemma_number": lemma_number + k, "init_lemma": parts[k] + ' -> ' + parts[k + 1], "included": included}
                    df.append(dct)
                lemma_number += len(parts) - 1
            else:
                dct = {"task_number": index, "lemma_number": lemma_number, "init_lemma": full_lemma, "included": included}
                df.append(dct)
                lemma_number += 1
            if included:
                lemma_old = full_lemma
    if not end:
        match = lemma_pattern.fullmatch(last_lemma)
        if match:
            full_lemma = match[1].strip()
            if full_lemma:
                dct = {"task_number": index, "lemma_number": lemma_number, "init_lemma": full_lemma, "included": False}
                df.append(dct)
    if len(df) and r'\boxed' in df[-1]["init_lemma"]:
        df[-1]["included"] = False
stdout.write('\n')

df = pd.DataFrame(df)
lemmas = list(check_hypotheses(df))
df['lemma'] = [lemma for lemma, included in lemmas]
df['included'] = [included for lemma, included in lemmas]
df.to_csv(output_file, index = False)
