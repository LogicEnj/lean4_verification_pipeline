import re
import numpy as np

logical_symbols = '↔→∨∧¬' #ascending order of priority
rel_symbols = '=<>≠∈≤≥≡∣'
terminate_symbols = logical_symbols + rel_symbols + '(){}+-*/,:^'
rel_with_multiple_symbols = ['Even', 'Odd', 'Nat.Prime', 'Prime']

def find_symb_outside_braces(symb, string):
    count1 = 0
    count2 = 0
    outside = -1
    for pos, char in enumerate(string):
        if char==symb and count1==0 and count2==0:
            outside = pos
            break
        else:
            match char:
                case '(':
                    count1 += 1
                case ')':
                    count1 -= 1
                case '{':
                    count2 += 1
                case '}':
                    count2 -= 1
    return outside

def find_all_symb_outside_braces(symb, string):
    count1 = 0
    count2 = 0
    outside = [-1]
    for pos, char in enumerate(string):
        if char==symb and count1==0 and count2==0:
            outside.append(pos)
        else:
            match char:
                case '(':
                    count1 += 1
                case ')':
                    count1 -= 1
                case '{':
                    count2 += 1
                case '}':
                    count2 -= 1
    outside.append(len(string))
    return outside

def contains_symb_of_str(string, str_of_symbols): #str_of_symbols does not contain various brackets
    contains = False
    count = 0
    for char in string:
        if char == '{':
            count += 1
        elif char == '}':
            count -=1
        elif char in str_of_symbols and count == 0:
            contains = True
            break
    return contains

def contains_rel_with_multiple_symbols(string):
    string = ' ' + string
    contains = False
    for sym in rel_with_multiple_symbols:
        possible_start_pos = find_all_symb_outside_braces(sym[0], string) [1 : -1]
        if not len(possible_start_pos):
            continue
        for pos in possible_start_pos:
            if re.search(rf'^{sym}\s', string[pos:]) and (string[pos-1].isspace() or string[pos-1] in logical_symbols):
                contains = True
                break
        if contains:
            break
    return contains

def add_missing_parentheses(expr):
    diff = expr.count('(') - expr.count(')')
    if diff<0:
        expr = '('*(-diff) + expr
    elif diff>0:
        expr = expr + ')'*diff
    return expr

def remove_outer_parentheses(expr):
    if not '(' in expr:
        expr = expr.replace(')', '')
    elif not ')' in expr:
        expr = expr.replace('(', '')
    empty_paren = re.compile(r'\(\s*\)', flags = re.M)
    while empty_paren.search(expr):
        expr = empty_paren.sub('', expr)

    expr = add_missing_parentheses(expr)

    string = np.array([char for char in '(' + expr + ')'], dtype = '<U1')
    result = np.zeros(string.size, dtype = '<U1')
    mask = np.ones(string.size, dtype = bool)
    not_spaces = np.array([not char.isspace() for char in string], dtype = bool)
    left_paren_indices = []

    for i, char in enumerate(string):
        if char == '(':
            left_paren_indices.append(i)
        elif char == ')':
            nonempty = mask[left_paren_indices[-1] + 1 : i] & not_spaces[left_paren_indices[-1] + 1 : i]
            if np.any(nonempty):
                local_mask = [ind for ind in range(left_paren_indices[-1] , i + 1) if mask[ind]]
                result[local_mask] = string[local_mask]
            mask[left_paren_indices[-1] : i + 1] = False
            left_paren_indices.pop()
            
    result = ''.join(result)
    balance = 0
    ind = 0
    for i, c in enumerate(result):
        if c == '(':
            balance += 1
        elif c == ')':
            balance -= 1
            if balance == 0:
                ind = i
                break

    if ind == len(result) - 1:
        result = result[1: -1]

    result = re.sub(r'\s+', ' ', result, flags = re.M)
    result = result.replace('( ', '(')
    result = result.replace(' )', ')')
    return result.strip()

def remove_types(hyp):
    result = []
    i = 0
    n = len(hyp)
    balance = 0
    while i < n:
        char = hyp[i]
        if balance > 0 and char == ':':
            j = i + 1
            while j < n and hyp[j] not in ',)':
                j += 1
            i = j - 1
        else:
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
            result.append(char)
        i += 1
    return remove_outer_parentheses(''.join(result))

def divide(symb, statement):
    pos = find_all_symb_outside_braces(symb, statement)
    return [remove_outer_parentheses(statement[pos[i] + 1 : pos[i + 1] ]) for i in range(len(pos) - 1)]

def rewrite_statement_with_names(statement):
    colon = find_symb_outside_braces(':',statement)
    
    left_side = remove_outer_parentheses(statement[: colon])
    if len(left_side) and left_side[0] != '(' and left_side[0] != '{':
        left_side = '(' + left_side + ')'
    conclusion = remove_outer_parentheses(statement[colon + 1 :])

    hypotheses_with_names = []
    intro_vars = []
    l_len = len(left_side)

    i = 0
    while i < l_len:
        while i < l_len and left_side[i] != '{' and left_side[i] != '(':
            i += 1
        if i == l_len:
            break
        start = i
        if left_side[i] == '{':
            balance = 1
            while i < l_len - 1 and balance:
                i += 1
                c = left_side[i]
                if c == '{':
                    balance += 1
                elif c == '}':
                    balance -= 1
            if balance:
                return [],[],None
            intro_vars.append(left_side[start: i + 1])
        else:
            balance = 1
            while i < l_len - 1 and balance:
                i += 1
                c = left_side[i]
                if c == '(':
                    balance += 1
                elif c == ')':
                    balance -= 1
            if balance:
                return [],[],None
            substatement = left_side[start + 1 : i]
            colon_idx = find_symb_outside_braces(':', substatement) + 1
            if colon_idx:
                subsubstatement = substatement[colon_idx : ]
                if contains_symb_of_str(subsubstatement, rel_symbols) or contains_rel_with_multiple_symbols(subsubstatement):
                    hyp_name = substatement[: colon_idx - 1].strip()
                    hyp = remove_outer_parentheses(subsubstatement)
                    hypotheses_with_names.append((hyp_name, hyp))
                else:
                    intro_vars.append('(' + substatement + ')')
            else:
                intro_vars.append('(' + substatement + ')')
        i += 1
    return intro_vars, hypotheses_with_names, conclusion

def rewrite_statement(statement):
    intro_vars, hypotheses_with_names, conclusion = rewrite_statement_with_names(statement)
    hypotheses = [hypothesis for name, hypothesis in hypotheses_with_names]
    return intro_vars, hypotheses, conclusion

def write_full_statement(intro_vars, hypotheses, conclusion, index):
    if len(intro_vars) + len(hypotheses):
        stat = ' '.join(intro_vars)
        for i, hypothesis in enumerate(hypotheses):
            stat += f' ({index}_{i}: {hypothesis})'
        stat += ' : '
    else:
        stat = ': '
    if conclusion:
        stat += conclusion
    else:
        stat += 'False'
    return stat

def get_vars(intro_vars):
    variables = set()
    for elem in intro_vars:
        if ':' in elem and '|' not in elem:
            names, typ = elem[1: -1].split(':', 1)
            typ = typ.strip()
            variables.update((name, typ) for name in re.findall(r'\S+', names))
    return variables

def replace_match(match):
    name = match.group(1).strip()
    variable = match.group(2).strip()
    return f': ∀ {variable}, {name} {variable} = '

def sub(match):
    if match[2]==':':
        return match[1] + ' :'
    else:
        return f'({match[1]}: Real) {match[2]}'

def sub_div(match):
    if match[4] or match[1].isalpha() or match[1]=='_':
        return match[0]
    else:
        return f'{match[1]} ({match[2]}: Real) / {match[3]}'

def sub_div2(match):
    if match[1].isalpha() or match[1]=='_':
        return match[0]
    else:
        return f'{match[1]} ({match[2]}: Real) / {match[3]} :='

def sub_abs(match):
    if match[1]:
        char = match[1]
        if char=='_' or char.isalnum():
            return match[0]
        else:
            return char + 'norm '
    else:
        return 'norm '

def add_type_to_func(typ):
    return lambda match: match[0] if match[1] in ('.', '_') else match[1] + typ + '.' + match[2]

def base_sub(match):
    if match[1] in terminate_symbols or match[1].isspace():
        return f'{match[1]}({match[2]} : Real)^'
    else:
        return match[0]

def correct_func(text):
    text = re.sub(r"\\n'?,?\s*'?", '\n', text, flags = re.M)
    text = re.sub(r'/-.*-/', '', text, flags = re.M | re.DOTALL) ##remove comment (can contains sorry)
    text = re.sub(r'\n *--.*','', text, flags = re.M) ##1:remove comments starting with --
    text = re.sub(r'^( *)· --.*\n *(.*)$', r'\1· \2', text, flags = re.M) ##2:replace comments starting with · -- by the next line
    text = re.sub(r' in ', ' ∈ ', text)

    text = text.replace('ℝ', 'Real')
    text = text.replace(r'\Real', 'Real')
    text = text.replace('ℕ', 'Nat')
    text = text.replace(r'\Nat', 'Nat')
    text = text.replace('ℤ', 'Int')
    text = text.replace(r'\Int', 'Int')
    text = text.replace('ℚ', 'Rat')
    text = text.replace(r'\Rat', 'Rat')
    text = text.replace('ℂ', 'Complex')
    text = text.replace('Float', 'Real')

    text = text.replace('Complex.abs', 'norm')
    text = re.sub(r'(.)?abs\s+', sub_abs, text, flags = re.M)


    #change all '= fun' for '∀ ... ,'
    pattern = re.compile(r':([^:=]+)=\s*fun\s+([^=]+)=>\s*')
    text = pattern.sub(replace_match, text)

    real = re.search(r':\s*Complex',text, flags = re.M) or re.search(r':\s*Real',text, flags = re.M)

    text = re.sub(r'(\d+\.\d+)\s*([^\d]?)', sub, text, flags = re.M)
    text = re.sub(r'(.)?(\d+\.?\d*)\s*/\s*(\d+\.?\d*)(\s*:)?', sub_div, text, flags = re.M)
    text = re.sub(r'(.)?(\d+\.?\d*)\s*/\s*(\d+\.?\d*)\s*:=', sub_div2, text, flags = re.M)
    if re.search(r':\s*Complex',text, flags = re.M) or re.search(r':\s*Real',text, flags = re.M):
        text = re.sub(r'(\D?)(\d+\.?\d*)\s*\^', base_sub, text, flags = re.M)

    text = re.sub(r'(.?)(sqrt)', add_type_to_func('Real'), text, flags = re.M)
    text = re.sub(r'([\s(])\.?I([\s)])', r'\1Complex.I\2', text, flags = re.M)

    return text
