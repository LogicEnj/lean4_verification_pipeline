import re
from qwen_shared import tokenizer, get_qwen_answer, filter_answer
import pandas as pd

br = re.compile(r'(_)?\{([^{}]*)\}', flags = re.M)
def br_sub(match):
    if match[1]:
        return match[0]
    else:
        return f' {match[2]} '

def rem_text(formulas):
    for formula in formulas:
        formula = formula[1: -1].replace('💲', r'\$')
        formula = re.sub(r'\\text\s*\{[^{}]*\}', '', formula, flags = re.M)
        formula = re.sub(r'\\mathbb\s*\{[^{}]*\}', '', formula, flags = re.M)
        formula = re.sub(r'\\mathbb\s*\S', '', formula, flags = re.M)
        formula = re.sub(r'\\[a-zA-Z]+\s*', '', formula, flags = re.M)
        formula_old = ''
        while br.search(formula) and formula_old != formula:
            formula_old = formula
            formula = br.sub(br_sub, formula)
        formula = re.sub(r'_\s+', '_', formula)
        yield formula

def get_vars(problem):
    problem = problem.replace(r'\$', '💲')
    problem = problem.replace('$$','$')
    problem = problem.replace(r'\[', '$')
    problem = problem.replace(r'\]', '$')
    problem = problem.replace(r'\(', '$')
    problem = problem.replace(r'\)', '$')
    problem = problem.replace(r'\begin{align*}', '$')
    problem = problem.replace(r'\end{align*}', '$')
    formula = '\n'.join(rem_text(re.findall(r'\$[^$]*\$', problem, flags = re.M)))
    variables = set(re.sub(r'\s+', '', result) for result in re.findall(r'\w+(?:\{[^{}]*\})?', formula, flags = re.M))
    variables2 = set()
    for var in variables:
        var = re.sub('_+', '_', var)
        var = re.sub('^_', '', var)
        var = re.sub('_$', '', var)
        if len(var):
            split = var.split('_')
            for i in range(len(split) - 1):
                left = re.sub(r'\{|\}','',split[i])
                right = split[i + 1]
                if len(left):
                    match = re.search(r'^\{[^{}]*\}', right)
                    if match:
                        end = match.end()
                        if end==3:
                            add = right[1]
                        else:
                            add = right[: end]
                    else:
                        end = 1
                        add = right[0]
                    variables2.update(left[:-1])
                    variables2.add(left[-1] + '_' + add)
                    variables2.update(right[end :])
            variables2.update(split[-1])        
    return list(set(var for var in variables2 if var[0].isalpha()))

def get_text_prompt_with_variables(modified_row):
    problem, variable, kind_of_stat = modified_row
    
    prompt = f"You are a math expert. For the following {kind_of_stat} and variable, determine the type of the variable and whether it is a mathematical constant."
    prompt += r"""
I will give you examples of types and definitions:

begin of example section

Statement:
Compute $a+b+c,$ given that $a,$ $b,$ and $c$ are the roots of \[\frac{1}{x} + 5x^2 = 6x - 24.\]
Variable:
b
Type:
$b\in\mathbb{R}$
Constant:
No

Statement:
What integer $x$ satisfies $\frac{1}{4}<\frac{x}{7}<\frac{1}{3}$?
Variable:
x
Type:
$x\in\mathbb{Z}$
Constant:
No

Statement:
Evaluate $(1+2i)6-3i$.
Variable:
i
Type:
$i\in\mathbb{C}$
Constant:
Yes

Statement:
If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction.
Variable:
f
Type:
$f\colon\mathbb{R}\to\mathbb{R}$
Constant:
No
end of example section

Put constant determination  between 
### Constant start
and
### Constant end
headers.

Put type of variable between 
### Type start
and
### Type end
headers.

"""
    prompt += f"Problem: \n{problem}\nVariable: \n{variable}\n"
    
    return prompt

def disjoint_union_of_variables(variables):
    for i, problem_variables in enumerate(variables):
        for variable in problem_variables:
            yield {"index": i, "variable": variable}

def get_by_pattern(pattern, descriptions):
    for description in descriptions:
        search_substring = re.search(pattern, description, flags = re.M|re.DOTALL)
        if search_substring:
            yield search_substring[1].strip()
        else:
            yield ''

def get_vars_data(df, num_of_statements):
    for index in range(num_of_statements):
        df_index = df[df['index'] == index]
        yield [re.sub(r'\{|\}','',var_name) for var_name in df_index['variable']], list(df_index['type']), list(df_index['constant']), list(df_index['full_description']), list(df_index['raw_output'])

def get_descriptions(statements_list, kind_of_stat, batch_size):
    if type(statements_list) != list:
        statements_list = list(statements_list)
    variables = [get_vars(statement) for statement in statements_list]
    df = pd.DataFrame(disjoint_union_of_variables(variables))
    if len(df):
        modify_row = lambda row: (statements_list[row['index']], row['variable'], kind_of_stat)
        raw_descriptions = get_qwen_answer(df, batch_size, modify_row, tokenizer, get_text_prompt_with_variables)
        descriptions = filter_answer(raw_descriptions)
        df['type'] = list(get_by_pattern(r'### Type start(.*)### Type end', descriptions))
        df['constant'] = [const=='Yes' for const in get_by_pattern(r'### Constant start(.*)### Constant end', descriptions)]
        df['full_description'] = descriptions
        df['raw_output'] = raw_descriptions
        vars_types_consts_fulls = list(get_vars_data(df, len(statements_list)))
    else:
        vars_types_consts_fulls = [[[],[],[],[],[]]]*len(statements_list)
    return variables, vars_types_consts_fulls
