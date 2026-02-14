import os
import re
import pandas as pd
import numpy as np
from correct_funcs import correct_func, rewrite_statement, get_vars, remove_outer_parentheses, contains_symb_of_str, divide, find_symb_outside_braces, logical_symbols, terminate_symbols, write_full_statement, remove_types, add_missing_parentheses
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder
import itertools
import ast
from sympy import symbols, parse_expr, Sum
from tokenize import TokenError
from sympy.core.sympify import SympifyError

def get_names_for_replace():
    import builtins
    import sympy
    names = set(dir(builtins))
    names.update(dir(sympy))
    return list(names)
repl_names = get_names_for_replace()

input_lemma_csv = get_input_file_or_folder('./config.yaml', 0)
input_csv = get_input_file_or_folder('./config.yaml', 1)
output_dir = get_output_file_or_folder('./config.yaml')
os.makedirs(output_dir, exist_ok = True)

print('input:', input_lemma_csv, input_csv)
print('output_dir:', output_dir)

del_spaces = re.compile(r'\s+', flags = re.M)
allowed = '+-*/()0123456789= '

def conclusion_list_to_statement(conclusion_list):
    if not conclusion_list:
        conclusion = 'False'
    elif len(conclusion_list) == 1:
        conclusion = conclusion_list.pop()
    else:
        conclusion = '∨'.join(f'( {conc} )' for conc in conclusion_list)
    return conclusion 

def divide_statement(hypotheses, conclusion, statement_type): #statement_type is 'lemma' or 'theorem'
    def _transfer_parts_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked):
        new_conclusions_unchecked = set()
        modified = False

        for conclusion in conclusions_unchecked:
            if conclusion.startswith('∃') or conclusion.startswith('∀') or not contains_symb_of_str(conclusion, logical_symbols):
                conclusions_checked.add(conclusion)
                modified = True
            else:
                equiv_pos = find_symb_outside_braces('↔', conclusion)
                if equiv_pos == -1:
                    rewrite_conclusion = divide('→', conclusion)
                    if len(rewrite_conclusion) == 1:
                        new_conclusions_unchecked.add(conclusion)
                    else:
                        hypotheses_unchecked.update(rewrite_conclusion[: -1])
                        new_conclusions_unchecked.add(rewrite_conclusion[-1])
                        modified = True
                else:
                    left = '(' + remove_outer_parentheses(conclusion[ : equiv_pos]) + ')'
                    right = '(' + remove_outer_parentheses(conclusion[equiv_pos + 1 : ]) + ')'
                    new_conclusions_unchecked.add(f'({left} → {right}) ∧ ({right} → {left})')
                    modified = True
        if modified:
            return hypotheses_checked, hypotheses_unchecked, conclusions_checked, new_conclusions_unchecked, True

        new_conclusions_unchecked.clear()
        for conclusion in conclusions_unchecked:
            rewrite_conclusion = divide('∨', conclusion)
            if len(rewrite_conclusion) == 1:
                new_conclusions_unchecked.add(conclusion)
            else:
                new_conclusions_unchecked.update(rewrite_conclusion)
                modified = True
        if modified:
            return hypotheses_checked, hypotheses_unchecked, conclusions_checked, new_conclusions_unchecked, True

        new_conclusions_unchecked.clear()
        for conclusion in conclusions_unchecked:
            if find_symb_outside_braces('∧', conclusion) == -1:
                match = re.search(r'^¬+', conclusion)
                if match:
                    num_neg = len(match[0])
                    modified = True
                    new_stat = remove_outer_parentheses(conclusion[num_neg : ])
                    if num_neg//2 :
                        hypotheses_unchecked.add(new_stat)
                    else:
                        new_conclusions_unchecked.add(new_stat)
                else:
                    new_conclusions_unchecked.add(conclusion)
            else:
                new_conclusions_unchecked.add(conclusion)
        if modified:
            return hypotheses_checked, hypotheses_unchecked, conclusions_checked, new_conclusions_unchecked, True

        new_hypotheses_unchecked = set()
        for hypothesis in hypotheses_unchecked:
            if hypothesis.startswith('∃') or hypothesis.startswith('∀') or not contains_symb_of_str(hypothesis, logical_symbols):
                hypotheses_checked.add(hypothesis)
                modified = True
            else:
                equiv_pos = find_symb_outside_braces('↔', hypothesis)
                if equiv_pos == -1:
                    if find_symb_outside_braces('→', hypothesis) == -1 and find_symb_outside_braces('∨', hypothesis) == -1:
                        rewrite_hypothesis = divide('∧', hypothesis)
                        if len(rewrite_hypothesis) == 1:
                            new_hypotheses_unchecked.add(hypothesis)
                        else:
                            new_hypotheses_unchecked.update(rewrite_hypothesis)
                            modified = True
                    else:
                        new_hypotheses_unchecked.add(hypothesis)
                else:
                    left = '(' + remove_outer_parentheses(hypothesis[ : equiv_pos]) + ')'
                    right = '(' + remove_outer_parentheses(hypothesis[equiv_pos + 1 : ]) + ')'
                    new_hypotheses_unchecked.update([left + '→' + right, right + '→' + left])
                    modified = True
        if modified:
            return hypotheses_checked, new_hypotheses_unchecked, conclusions_checked, conclusions_unchecked, True

        new_hypotheses_unchecked.clear()
        for hypothesis in hypotheses_unchecked:
            match = re.search(r'^¬+', hypothesis)
            if match:
                num_neg = len(match[0])
                modified = True
                new_stat = remove_outer_parentheses(hypothesis[num_neg : ])
                if num_neg//2 :
                    conclusions_unchecked.add(new_stat)
                else:
                    new_hypotheses_unchecked.add(new_stat)
            else:
                new_hypotheses_unchecked.add(hypothesis)
        if modified:
            return hypotheses_checked, new_hypotheses_unchecked, conclusions_checked, conclusions_unchecked, True

        return hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked, False

    def _transfer_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked):
        transfer = True
        while transfer:
            hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked, transfer = _transfer_parts_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked)
        return hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked

    def _divide_lemma_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked, lemmas):
        hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked = _transfer_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked)

        conj = False
        for conclusion in conclusions_unchecked:
            if find_symb_outside_braces('∧', conclusion) != -1:
                conj = True
                break

        disj = False
        for hypothesis in hypotheses_unchecked:
            if find_symb_outside_braces('∨', hypothesis) != -1:
                disj = True
                break

        if conj or disj or len(hypotheses_unchecked):
            if conj:
                expr = 'itertools.product(' + ', '.join(str(divide('∧', conclusion)) for conclusion in conclusions_unchecked) + ')'
                for subconclusion in eval(expr):
                    _divide_lemma_(hypotheses_checked.copy(), hypotheses_unchecked.copy(), conclusions_checked.copy(), set(subconclusion), lemmas)
            if disj:
                expr = 'itertools.product(' + ', '.join(str(divide('∨', hypothesis)) for hypothesis in hypotheses_unchecked) + ')'
                for subhypothesis in eval(expr):
                    _divide_lemma_(hypotheses_checked.copy(), set(subhypothesis), conclusions_checked.copy(), conclusions_unchecked.copy(), lemmas)

            for i, hypothesis in enumerate(hypotheses_unchecked):
                impls = divide('→', hypothesis)
                if len(impls) > 1:
                    hypotheses_unchecked_without = hypotheses_unchecked.copy()
                    hypotheses_unchecked_without.discard(hypothesis)
                    hypotheses_unchecked_without.add(impls[-1])
                    _divide_lemma_(hypotheses_checked.copy(), hypotheses_unchecked_without.copy(), conclusions_checked.copy(), conclusions_unchecked.copy(), lemmas)
                    hypotheses_unchecked_without.discard(impls[-1])
                    for impl in impls[: -1]:
                        conclusions_unchecked.add(impl)
                        _divide_lemma_(hypotheses_checked.copy(), hypotheses_unchecked_without.copy(), conclusions_checked.copy(), conclusions_unchecked.copy(), lemmas)
                        conclusions_unchecked.discard(impl)

        else:
            hypotheses_checked = hypotheses_checked | hypotheses_unchecked
            conclusions_checked = conclusions_checked | conclusions_unchecked
            if not hypotheses_checked & conclusions_checked and 'True' not in conclusions_checked and 'False' not in hypotheses_checked:
                hypotheses_checked.discard('True')
                conclusions_checked.discard('False')
                lemmas.append( (list(hypotheses_checked), list(conclusions_checked)) )

    def _divide_theorem_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked, lemmas):
        hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked = _transfer_(hypotheses_checked, hypotheses_unchecked, conclusions_checked, conclusions_unchecked)
        hypotheses_checked = hypotheses_checked | hypotheses_unchecked
        conclusions_checked = conclusions_checked | conclusions_unchecked
        if hypotheses_checked & conclusions_checked or 'True' in conclusions_checked or 'False' in hypotheses_checked:
            lemmas.append( ([], 'True') )
        else:
            hypotheses_checked.discard('True')
            conclusions_checked.discard('False')
            lemmas.append( (list(hypotheses_checked), conclusion_list_to_statement(conclusions_checked)) )

    lemmas = []
    if statement_type == 'lemma':
        _divide_ = _divide_lemma_
    elif statement_type == 'theorem':
        _divide_ = _divide_theorem_
    _divide_(set(), set(hypotheses), set(), {conclusion}, lemmas)
    return lemmas

def prepare_for_sympy(expression, fun_dict):
    def sqrt_sub(expr):
        def _sub_(match):
            sqrts = match[1].replace('√','sqrt(')
            fin = ')' * len(match.group(1))
            other = match[2].strip()
            ind = 0
            if other.startswith('('):
                sqrts = sqrts[:-1]
                balance = 1
                ind = 1
                while ind < len(other) and balance:
                    match other[ind]:
                        case '(':
                            balance += 1
                        case ')':
                            balance -= 1
                    ind += 1
                fin = fin[:-1]
            else:
                while ind < len(other) and other[ind] not in terminate_symbols:
                    ind += 1
            if ind:
                ret_str = sqrts + other[: ind] + fin + other[ind :]
            else:
                ret_str = match[0]
            return ret_str
        return re.sub(r'(√+)([^√]*)', _sub_, expr, flags = re.M)

    def add_parens_to_func(func_name, expr, nat):
    #replace <sympy_name> arg_1 ... arg_s by sympy_name(arg_1,...,arg_s) if nat==False
    #replace Nat.<sympy_name> arg_1 ... arg_s by max(0,sympy_name(arg_1,...,arg_s)) if nat==True
        def _add_parens_(sympy_name, func_arity, expr, search_expr, open_expr, close_expr):
            n = len(search_expr)
            args = np.empty(func_arity, dtype = object)
            ind = expr.find(search_expr)
            left = True
            result = ''
            while ind != -1:
                len_expr = len(expr)
                end = n + ind
                right_cond = end < len_expr and expr[end].isspace()
                left_cond = ind == 0 or expr[ind - 1].isspace() or expr[ind - 1] in terminate_symbols
                append = True
                if left and right_cond and left_cond:
                    for i in range(func_arity):
                        while end < len_expr and expr[end].isspace():
                            end += 1
                        if end == len_expr:
                            append = False
                            break
                        start = end
                        if expr[start] == '(':
                            balance = 1
                            while end < len_expr and balance:
                                end += 1
                                match expr[end]:
                                    case '(':
                                        balance += 1
                                    case ')':
                                        balance -= 1
                            end += 1
                            if end > len_expr:
                                append = False
                                break
                            args[i] = expr[start + 1: end - 1]
                        elif expr[start] in terminate_symbols:
                            append = False
                            break
                        else:
                            while end < len_expr and not expr[end].isspace() and not expr[end] in terminate_symbols:
                                end += 1
                            args[i] = expr[start: end]
                    if append:
                        result +=  expr[: ind] + open_expr + ', '.join([_add_parens_(sympy_name, func_arity, arg, search_expr, open_expr, close_expr) for arg in args]) + close_expr
                        left = True
                    else:
                        result += expr[: end]
                        left = False
                else:
                    result += expr[: end]
                    left = False
                expr = expr[end :]
                ind = expr.find(search_expr)
            return (result + expr).strip()
        
        sympy_name, func_arity = func_name
        if nat:
            search_expr = 'Nat.' + sympy_name
            open_expr = 'max(0, ' + sympy_name + '('
            close_expr = '))'
        else:
            search_expr = sympy_name
            open_expr = sympy_name + '('
            close_expr = ')'
        return _add_parens_(sympy_name, func_arity, expr, search_expr, open_expr, close_expr)

    def logb_parse(expr):
        find_logb = expr.find('logb')
        while find_logb >= 0:
            start = expr[: find_logb + 3] + '('
            leng = len(expr) - 1
            balance = 1
            end_ind = find_logb + 4
            while balance and end_ind < leng:
                end_ind += 1
                match expr[end_ind]:
                    case '(':
                        balance += 1
                    case ')':
                        balance -= 1
            end = expr[end_ind : ]
            args = expr[find_logb + 5 : end_ind]
            delimeter = find_symb_outside_braces(',', args)
            expr = start + args[delimeter + 1 : ] + ',' + args[: delimeter] + end
            find_logb = expr.find('logb')
        return expr

    def change(string):
        string = re.sub(r'\s+range\s+', ' Icc 0 ', string.replace('Finset.', ''), flags = re.M)
        string = re.sub(r'\s+,', ',', string, flags = re.M)
        trans_dct = {ord(sym): ' ' for sym in terminate_symbols if sym not in '()'}
        def arg_remain(string): ##first symbol of string is not a space
            n = len(string)
            if string[0] == '(':
                i = 1
                balance = 1
                while i < n and balance:
                    match string[i]:
                        case '(':
                            balance += 1
                        case ')':
                            balance -= 1
                    i += 1
                if i == n and balance:
                    return n, True
            else:
                i = re.search(r'[^\s()]*', string, flags = re.M).end()
            return i, False
            
        def find_(string):
            old = string
            y = re.search(r'∑\s+(\w+)\s+in\s+Icc\s+', string, flags = re.M)
            if y:
                ind = y[1]
                string = string[y.end():]
                pos, not_correct = arg_remain(string)
                if not_correct:
                    return '', old, True
                arg1 = string[:pos]
                string = string[pos:].strip()

                pos, not_correct = arg_remain(string)
                if not_correct:
                    return '', old, True
                arg2 = string[:pos]
                string = string[pos:].strip()
                if arg2.endswith(','):
                    arg2 = arg2[:-1]

                if string.startswith('∑'):
                    arg3, string, not_correct = find_(string)
                    if not_correct:
                        return '', old, True
                else:
                    pos, not_correct = arg_remain(string.translate(trans_dct))
                    if not_correct:
                        return '', old, True
                    else:
                        arg3 = string[:pos]
                        string = string[pos:].strip()
                return f"Sum({arg3}, ({ind}, {arg1}, {arg2}))" , string, False
            else:
                return '', old, False
        find = string.find('∑')
        if find < 0:
            return string
        else:
            left, right, _ = find_(string[find:])
            return string[:find] + left + right

    def substitute_name(match):
        sym1 = match[1].strip()
        sym2 = match[3].strip()
        if sym1 in terminate_symbols and sym2 in terminate_symbols and sym2 != '(': #in particular, sym1 and sym2 can be empty strings
            return f'{match[1]}{match[2]}_repl{match[3]}'
        else:
            return match[0]

    if not '(' in expression:
        expression = expression.replace(')', '')
    elif not ')' in expression:
        expression = expression.replace('(', '')
    add_missing_parentheses(expression)
    
    expression = remove_types(expression)

    abs_pattern = re.compile(r'\|([^|{}]*)\|', flags = re.M)
    ceiling_pattern = re.compile(r'⌈([^⌈⌉]*)⌉', flags = re.M)
    floor_pattern = re.compile(r'⌊([^⌊⌋]*)⌋', flags = re.M)
    for func in [(abs_pattern, 'abs'), (ceiling_pattern, 'ceiling'), (floor_pattern, 'floor')]:
        pattern, func_name = func
        while pattern.search(expression):
            expression = pattern.sub(rf'{func_name}(\1)', expression)
    while expression.count('‖') > 1:
        expression = re.sub(r'‖([^‖]*)‖',r'abs(\1)', expression)
    expression = sqrt_sub(expression)
    
    expression = expression.replace('Real.sqrt', 'sqrt').replace('norm', 'abs').replace('Int.ceil', 'ceiling').replace('Int.floor', 'floor').replace('Real.logb', 'logb').replace('Real.log', 'log')
    for func_name in fun_dict:
        expression = add_parens_to_func(func_name, expression, False)
    expression = logb_parse(expression)

    expression = expression.replace('Nat.ceil', 'Nat.ceiling')
    for func_name in [('ceiling', 1), ('floor', 1)]:
        expression = add_parens_to_func(func_name, expression, True)
    
    expression = expression.replace('≤','<=').replace('≥','>=').replace('≠','!=').replace('∈',' in ').replace('Complex.I', 'I').replace('^', '**')
    expression = re.sub(r'\.eval\s+(\d+)(\.\d+)?', r'(\1\2)', expression)
    expression = re.sub(r'\s*\.I\s*([\s)}])', r' I\1',expression)
    expression = re.sub(r'\s*\.I\s*$', r' I',expression)

    expression = change(expression)
    for name in repl_names:
        expression = re.sub(rf'(.?)({name})(\s*.?)', substitute_name, expression, flags = re.M | re.DOTALL)
    expression = remove_outer_parentheses(expression)

    ret_expr = expression
    if '(' in ret_expr:
        try:
            if '=' in ret_expr and not '!=' in ret_expr and not '<=' in ret_expr and not '>=' in ret_expr:
                lhs, rhs = (remove_outer_parentheses(expr) for expr in ret_expr.split('=', 1))
                ret_expr = f"{ast.unparse(ast.parse(lhs))} = {ast.unparse(ast.parse(rhs))}"
            else:
                ret_expr = ast.unparse(ast.parse(ret_expr))
        except (SyntaxError, TypeError, AttributeError) as e:
            print('prepare_for_sympy: ', e)
            print('error expression: ', ret_expr)
            ret_expr = re.sub(r'\s+', '', expression, flags = re.M)
    return ret_expr

def unnecessary_hypothesis(hyp, symb, fun_dict):
    if hyp.count('/') > 2 or hyp.count('+') > 0 or hyp.count('*') > 0 or hyp.count(symb) != 1:
        return False
    hyp1, hyp2 = (prepare_for_sympy(part, fun_dict) for part in hyp.split(symb)) 
    if hyp1.count(' ') > 0 or hyp2.count(' ') > 0:
        return False
    check_num = re.compile(r'^-?\d+([/\.]\d+)?$')
    num1 = bool(check_num.fullmatch(hyp1))
    num2 = bool(check_num.fullmatch(hyp2))
    if num1 == num2:
        return False
    if num2 and not re.search(r'\W', hyp1) or num1 and not re.search(r'\W', hyp2):
        return True
    return False

def get_definitions(file): #we believe that there are no other constructions, so the header ends in 'def'
    def add_def(file):
        for part in re.split(r'\sdef\s', ' ' + file[:file.find('theorem')], flags = re.M)[1:]:
            part = part.strip()
            name = re.search(r'^[^\s(:]*', part)[0]
            if name:
                yield 'def ' + part + '\n\n', name
    return list(add_def(file))

def get_opens(file): #to add 'open Real' to header
    def add_open(file):
        for part in file.split('\n'):
            part = part.strip()
            if part.startswith('open'):
                yield del_spaces.sub(' ', part)

    return list(add_open(file))

def check_for_header(lemma_def, def_header):
    for definition, name in lemma_def:
        if not any(header_name == name for _, header_name in def_header):
            def_header.append((definition, name))
                #in the future, there will be code that does the comparison and, if they are not the same not including ' ', there will be a query to the Lean client that includes the following:
                #f'import Mathlib \n def f {elem_part} \n def g {definition_part} \n example {the needed variables with their types}: f x = g x := by simp [f, g] \n try linarith \n try norm_num \n try simp_all'
                #and if the query fails, Kimina should autoformalize the part better with the 'elem_part' in mind. Right now, we just replace definition by elem and hope for the best
    return def_header

def check_unification(A, B):
    if del_spaces.sub('', A) == del_spaces.sub('', B):
        return True
    try:
        expr1 = parse_expr(A, local_dict={'Sum': Sum})
        expr2 = parse_expr(B, local_dict={'Sum': Sum})
        return expr1.equals(expr2)
    except (SyntaxError, TypeError, AttributeError, TokenError, SympifyError) as e:
        print('check_unification_error: ', e)
        print(f'A, B: \n{A}\n{B}')
        return False

def find_closest(hypothesis, hyp_vars, we_work, we_work_vars, fun_dict):
    if hypothesis.count('=') == 1:
        A, B = (prepare_for_sympy(part, fun_dict) for part in hypothesis.split('='))
        if check_unification(A, B):
            return ''
        for elem, variables in zip(we_work, we_work_vars):
            if not variables.issubset(hyp_vars) or elem.count('=') != 1:
                continue
            C, D = (prepare_for_sympy(part, fun_dict) for part in elem.split('='))
            if check_unification(A, C) and check_unification(B, D) or check_unification(A, D) and check_unification(B, C):
                return elem
    return 'not possible'

def is_numerical_formula(hyp):
    hyp = hyp.replace('$','').replace('^', '**').strip()
    if hyp.count('=') == 1 and all(c in allowed for c in hyp):
        try:
            expr1, expr2 = ( parse_expr(part.strip(), evaluate = True) for part in hyp.split('=', 1) )
        except:
            return False
        return expr1 == expr2
    return False

def compare(texts, lean_list):
    def get_parts(lean_list):
        for lean in lean_list:
            lean = lean.replace('Real', '').replace('Int', '').replace('Nat', '').replace('Rat', '').replace('Complex', '').replace(':', '').replace('(', '').replace(')', '')
            yield del_spaces.sub('',lean).split('∧')

    def append_stat(texts, parts):
        for text in texts:
            if is_numerical_formula(text) and not any(del_spaces.sub('',text) in part for part in parts):
                yield text

    parts = list(get_parts(lean_list))
    return list(append_stat(texts, parts))

def check_for_missing_formula(hypotheses, conclusion, lemma_eng):
    lemma_eng = lemma_eng.replace('$', '')
    if '->' in lemma_eng:
        part1, concl = (part.strip() for part in lemma_eng.split('->', 1))
        hyp_list = [hyp.strip() for hyp in part1.split(r'\land')]
        hypotheses.extend(compare(hyp_list, hypotheses))
        concl_append = compare([concl], [conclusion])
        if concl_append:
            conclusion += ' ∧ ' + concl_append[0]
    else:
        concl_list = [hyp.strip() for hyp in lemma_eng.split(r'\land')]
        concl_append = compare(concl_list, [conclusion])
        if concl_append:
            conclusion  += ' ∧ ' + ' ∧ '.join(concl_append)
    return hypotheses, conclusion

def get_potential_variables(text):
    text = text.translate({ord(c): ' ' for c in terminate_symbols})
    return set(re.findall(r'\S+', text))

def get_functions(variables):
    for name, typ in variables:
        pos = typ.rfind('→')
        if pos>-1:
            arity = typ.count('→') + typ[ :pos].count('×')
            yield (name, arity)

def power_equation(hypotheses):
    for hypothesis in hypotheses:
        hypothesis = del_spaces.sub('',hypothesis)
        if hypothesis.startswith('∃') or hypothesis.startswith('∀') or contains_symb_of_str(hypothesis, logical_symbols) or '=' not in hypothesis:
            yield ()
            continue
        eq_parts = [remove_types(part) for part in hypothesis.split('=', 1)]
        exp_positions = [find_symb_outside_braces('^', part) for part in eq_parts]
        if any(pos == -1 for pos in exp_positions):
            yield ()
            continue
        base1, base2 = (remove_outer_parentheses(part[:pos]) for part, pos in zip(eq_parts, exp_positions))
        if base1 != base2 or not re.fullmatch(r'\d+\.?\d*', base1):
            yield ()
            continue
        pow1, pow2 = (remove_outer_parentheses(part[pos+1:]) for part, pos in zip(eq_parts, exp_positions))
        yield (base1, pow1, pow2)

def conclusion_equation(conclusion, hypotheses_pow_equations):
    conclusion = del_spaces.sub('',conclusion)
    if not any(conclusion.startswith(sym) for sym in '∃∀') and not contains_symb_of_str(conclusion, logical_symbols) and '=' in conclusion:
        concl_parts = tuple(remove_outer_parentheses(part) for part in conclusion.split('=', 1))
        for i, equ in enumerate(hypotheses_pow_equations):
            if equ and set(equ[1:]) == set(concl_parts):
                yield i


theorems = pd.read_csv(input_csv)
lemmas = pd.read_csv(input_lemma_csv)

theorems['header'] = None
theorems['equivalence'] = False
theorems['direct'] = None
theorems['converse'] = None

lemma_lookup = re.compile(r'theorem.*:=', flags = re.M | re.DOTALL)
header_start = 'import Mathlib\n\nset_option maxHeartbeats 0\n\n'
for task_number, row in theorems.iterrows():
    theorem = row['theorem']
    match = lemma_lookup.search(theorem)
    if not match:
        theorems.at[task_number, 'correct'] = False
        continue
    theorem_match = re.search(r'theorem[^{(:]*', theorem)
    begin_theorem_stat = theorem_match.end()
    end_theorem_stat = match[0].find(':=') + theorem_match.start()
    th_intro_vars, th_hypotheses, th_conclusion = rewrite_statement(theorem[begin_theorem_stat : end_theorem_stat])
    if th_conclusion is None:
        theorems.at[task_number, 'correct'] = False
        continue

    def_header = get_definitions(theorem)
    open_header = get_opens(theorem)

    equivalence = find_symb_outside_braces('↔', th_conclusion)
    if equivalence > -1:
        theorems.at[task_number, 'equivalence'] = True
        right_side = remove_outer_parentheses(th_conclusion[equivalence + 1 : ])
        left_side = remove_outer_parentheses(th_conclusion[ : equivalence])
        theorems.at[task_number, 'converse'] = theorem_match[0] + ' ' + write_full_statement(th_intro_vars, th_hypotheses, f'({right_side}) → ({left_side})', 'h0') + ' := by\n'
        th_hypotheses.append(left_side)
        th_conclusion = right_side

    th_hypotheses, th_conclusion = divide_statement(th_hypotheses, th_conclusion, 'theorem')[0]
    if th_conclusion == 'True':
        theorems.at[task_number, 'direct'] = theorem_match[0] + ': True := by\n'
        theorems.at[task_number, 'header'] = header_start
        continue
    theorems.at[task_number, 'direct'] = theorem_match[0] + ' ' + write_full_statement(th_intro_vars, th_hypotheses, th_conclusion, 'h0') + ' := by\n'
    th_variables = get_vars(th_intro_vars)
    fun_dict = {('abs', 1), ('sqrt', 1), ('ceiling', 1), ('floor', 1), ('sin', 1), ('cos', 1), ('arcsin', 1), ('arccos', 1), ('cot', 1), ('ln', 1), ('log', 1), ('logb', 2)}
    fun_dict.update(get_functions(th_variables))
    th_variables = set(name for name, typ in th_variables)
    vars_of_theorem_hypotheses = [get_potential_variables(hypothesis)&th_variables for hypothesis in th_hypotheses]
    th_power_equations = list(power_equation(th_hypotheses))
    we_work = th_hypotheses.copy()
    we_work_sympy = [prepare_for_sympy(hyp, fun_dict) for hyp in we_work]
    we_work_vars = vars_of_theorem_hypotheses.copy()
    lemma_number = 1
    for _, row in lemmas.iterrows():
        if row['task_number'] != task_number:
            continue
        file = correct_func(row['formalization'])
        match = lemma_lookup.search(file)
        if not match:
            continue
        solution = match[0]
        solution = solution[re.search(r'theorem[^{(:]*', solution).end() : solution.find(':=')].strip()
        intro_vars, hypotheses, conclusion = rewrite_statement(solution)
        if conclusion is None:
            continue
        hypotheses, conclusion = check_for_missing_formula(hypotheses, conclusion, row['lemma'])

        start = ' '.join(intro_vars) + ' '
        variables_with_types = get_vars(intro_vars)
        fun_dict.update(get_functions(variables_with_types))
        variables = set(name for name, typ in variables_with_types)

        def_header = check_for_header(get_definitions(file), def_header)
        open_header.extend((op for op in get_opens(file) if op not in open_header))
        struct_lemma = '-- ' + row['lemma'] + '\n\n'
        header = header_start + '\n'.join(open_header) + '\n' + ''.join(definition for definition, _ in def_header)
        file_start = header + 'theorem LEMMA ' + start
        fact_start = header + 'theorem FACT ' + start

        fact_th_hyp_str = ''
        j = 0
        file_th_hyp = []
        file_pow_eq = []
        for hyp, hyp_vars, hyp_pow in zip(th_hypotheses, vars_of_theorem_hypotheses, th_power_equations):
            if hyp_vars.issubset(variables):
                fact_th_hyp_str += f'(F_{j}: {hyp}) '
                j += 1
                if not del_spaces.sub('',hyp) in del_spaces.sub('',file):
                    file_th_hyp.append(hyp)
                    file_pow_eq.append(hyp_pow)
        file_th_hyp_str = ''.join(f'(F_{j}: {hyp}) ' for j, hyp in enumerate(file_th_hyp))

        for hypotheses, conclusion_list in divide_statement(hypotheses, conclusion, 'lemma'):
            power_equations = list(power_equation(hypotheses))
            power_equations.extend(file_pow_eq)
            not_pow_eq = True
            if any(power_equations):
                indices = [list(conclusion_equation(conclusion, power_equations)) for conclusion in conclusion_list]
                if any(indices):
                    not_pow_eq = False
                    concl_ind = min(i for i, concl_indices in enumerate(indices) if concl_indices)
                    conclusion = conclusion_list[concl_ind]
                    hypothesis_ind = min(indices[concl_ind])
                    if hypothesis_ind < len(hypotheses):
                        hypothesis = hypotheses[hypothesis_ind]
                    else:
                        hypothesis = file_th_hyp[hypothesis_ind - len(hypotheses)]
                    hypotheses = [hypothesis]
                    file_th_hyp_local = ''
                    file_hyp = f'(h: {hypothesis}) '
                    if any((name, 'Real') in variables_with_types for name in get_potential_variables(conclusion)):
                        hyp_add_real = re.sub(r'\^\s*(\d+\.?\d*)', r'^(\1: Real)', hypothesis, flags = re.M|re.DOTALL)
                        base_real = f'({power_equations[hypothesis_ind][0]}: Real)'
                        file_end = f'''
  have l1 {start} (h1: {hypothesis}) : {hyp_add_real} := by linarith
  have l2 {start} (h2: {hyp_add_real}) : {conclusion} := by
    have v1: 0 < {base_real} := by simp
    have v2: {base_real} ≠ 1 := by simp
    refine (Real.rpow_right_inj v1 v2).1 h2
  solve_by_elim (maxDepth := 20)'''
            if not_pow_eq:
                file_hyp = ''.join(f'(h_{i}: {hypothesis}) ' for i, hypothesis in enumerate(hypotheses))
                conclusion = conclusion_list_to_statement(conclusion_list)
                file_th_hyp_local = file_th_hyp_str
                file_end = ' sorry'
            file = struct_lemma + correct_func(file_start + file_hyp + file_th_hyp_local + ': ' + conclusion + ' := by' + file_end)
            with open(os.path.join(output_dir, f'thm_{task_number}_{lemma_number}.lean'), "w", encoding = "utf-8") as f:
                f.write(file)

            j = 0
            for hypothesis in hypotheses:
                hyp_sympy = prepare_for_sympy(hypothesis, fun_dict)
                if hyp_sympy in we_work_sympy or unnecessary_hypothesis(hypothesis, '=', fun_dict):
                    continue
                hyp_vars = get_potential_variables(hypothesis)&variables
                elem = find_closest(hypothesis, hyp_vars, we_work, we_work_vars, fun_dict)
                if elem != 'not possible':
                    if elem:
                        elem = f'(F_elem: {elem}) '
                    fact = struct_lemma + correct_func(fact_start + elem + fact_th_hyp_str + ': ' + hypothesis + ' := by sorry')
                    with open(os.path.join(output_dir, f'fact_{task_number}_{lemma_number}_{j}.lean'), 'w', encoding = 'utf-8') as f:
                        f.write(fact)
                    j += 1
                we_work.append(hypothesis)
                we_work_sympy.append(hyp_sympy)
                we_work_vars.append(hyp_vars)

            conclusion_sympy = prepare_for_sympy(conclusion, fun_dict)
            if conclusion_sympy not in we_work_sympy:
                we_work.append(conclusion)
                we_work_sympy.append(conclusion_sympy)
                we_work_vars.append(get_potential_variables(conclusion)&variables)

            lemma_number += 1

    header = header_start + '\n'.join(open_header) + '\n\n' + ''.join(definition for definition, _ in def_header)
    theorems.at[task_number, 'header'] = header
theorems.to_csv(os.path.join(output_dir, 'theorems.csv'), index = False)
