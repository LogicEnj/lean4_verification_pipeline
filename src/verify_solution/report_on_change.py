import copy
from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
import requests
import pandas as pd
import os
import re


def read_file_content(file_path, default="NO FILE"):
    """Read file content or return default if file does not exist."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return default

def tactics_num(directory):
    tac_pattern = re.compile(r'^task-\d+-(\d+).lean$')
    for filename in os.listdir(directory):
        match = tac_pattern.fullmatch(filename)
        if match and os.path.isfile(os.path.join(directory, filename)):
            yield int(match[1])

######################################## LOAD FILE NAMES #######################

old_config_file = './config_old.yaml'
new_config_file = './config.yaml'

output_file = get_output_file_or_folder(new_config_file)
csv_str_lemmas = {}
dir_unproved = {}
dir_proved = {}
dir_final = {}

dir_unproved['old'] = get_input_file_or_folder(old_config_file, 1)
csv_str_lemmas['old'] = get_input_file_or_folder(old_config_file, 0)
dir_proved['old'] = get_input_file_or_folder(old_config_file, 2)
dir_final['old'] = get_input_file_or_folder(old_config_file, 3)

dir_unproved['new'] = get_input_file_or_folder(new_config_file, 1)
csv_str_lemmas['new'] = get_input_file_or_folder(new_config_file, 0)
dir_proved['new'] = get_input_file_or_folder(new_config_file, 2)
dir_final['new'] = get_input_file_or_folder(new_config_file, 3)

print('input_old:', csv_str_lemmas['old'], dir_unproved['old'], dir_proved['old'], dir_final['old'])
print('input_new:', csv_str_lemmas['new'], dir_unproved['new'], dir_proved['new'], dir_final['new'])
print('output:', output_file)

####################################### READ RESULTS ###################################
def get_result(directory):
    pattern = re.compile(r'^task-(\d+)-\d+\.err$')
    dct_correct = {}
    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        fullpath = os.path.join(directory, filename)
        if match and os.path.isfile(fullpath):
            task_number = int(match[1])
            if dct_correct.get(task_number, False):
                continue
            with open(fullpath, 'r', encoding = 'utf-8') as file:
                dct_correct[task_number] = file.read().endswith('TRUE')

    correct = [key for key, val in dct_correct.items() if val]
    incorrect = [key for key, val in dct_correct.items() if not val]
    return {'correct': correct, 'incorrect': incorrect}

def lemma_indices(directory):
    pattern = re.compile(r'^thm_(\d+)_(\d+)\.lean$')
    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        if match and os.path.isfile(os.path.join(directory, filename)):
            yield int(match[1]), int(match[2])

def fact_indices(directory):
    pattern = re.compile(r'^fact_(\d+)_(\d+)_(\d+)\.lean$')
    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        if match and os.path.isfile(os.path.join(directory, filename)):
            yield int(match[1]), int(match[2]), int(match[3])

def facts_and_lemmas_for_task(task_number, lemma_inds, fact_inds):
    for task_ind, lemma_ind in lemma_inds:
        if task_ind != task_number:
            continue
        for task_ind2, lemma_ind2, fact_ind in fact_inds:
            if task_ind2 != task_number or lemma_ind2 != lemma_ind:
                continue
            yield f'Fact {lemma_ind}_{fact_ind}', f'fact_{task_ind}_{lemma_ind}_{fact_ind}.lean'
        yield f'Lemma {lemma_ind}', f'thm_{task_ind}_{lemma_ind}.lean'

num_of_tactics = {version: max(tactics_num(dir_final[version])) + 1 for version in ('old','new')}

evaluation_info = {}
for version in ('old', 'new') :
    evaluation_info[version] = get_result(dir_final[version])

######################################## READ LEMMAS AND THEOREMS #######################

reports = {}

for version in ('old', 'new'):
    lemmas = pd.read_csv(csv_str_lemmas[version])
    df = pd.read_csv(os.path.join(dir_proved[version], 'theorems.csv'))
    print(len(df), "records")

    lemma_inds = sorted(list(lemma_indices(dir_unproved[version])))
    fact_inds = sorted(list(fact_indices(dir_unproved[version])))

    data = []
    for index, row in df.iterrows():
        task_number = index
        info = {}
        info['problem'] = row['problem'] + f'<br><br>\n Problem {task_number}'
        info['init_structured_solution'] = row['structured_solution']
        info['structured_solution'] = '\n'.join(_row_['lemma'] for _, _row_ in lemmas.iterrows() if _row_['task_number'] == task_number)
        for tactic_ind in range(num_of_tactics[version]):
            final_path = os.path.join(dir_final[version], f"task-{task_number}-{tactic_ind}.lean") 
            info[f'final_theorem_{tactic_ind}'] = read_file_content(final_path)
            full_path_to_error_message_file = final_path[:-5] + '.err'
            info[f'error_{tactic_ind}'] = read_file_content(full_path_to_error_message_file)
            info[f'final_result_{tactic_ind}'] = info[f'error_{tactic_ind}'].endswith('TRUE')
        info['lemmas'] = [{'name': name, 'unproved': read_file_content(os.path.join(dir_unproved[version], filename)), 'proved': read_file_content(os.path.join(dir_proved[version], filename)), 'error': read_file_content(os.path.join(dir_proved[version], filename[:-5]+'.err'))}
                          for name, filename in facts_and_lemmas_for_task(task_number, lemma_inds, fact_inds)]
        data.append(copy.deepcopy(info))

    reports[version] = copy.deepcopy(data)

assert len(reports['old']) == len(reports['new'])


############################################## display report as HTML

def escape_html(text):
    if not isinstance(text, str):
        return ''
    return (text.replace('&', '&amp;')
            .replace('"', '&quot;')
            .replace("'", '&#x27;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            )

# Start building the HTML
html = '''
<html>
<head>
<style>
  table {
    border-collapse: collapse;
    width: 100%;
    font-family: Arial, sans-serif;
  }
  th, td {
    border: 1px solid #999;
    padding: 6px;
    vertical-align: top;
    word-wrap: break-word;
    max-width: 300px;
  }
  pre {
    margin: 0;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .structured-solution {
    white-space: pre-line;
  }
  .arrow {
    background-color: yellow;
  }
  .lemma-green {
    background-color: #ccffcc;
  }
  .lemma-red {
    background-color: #ff9999;
  }
  .problem-row {
    background-color: #f2f2f2;
  }
  .lemma-cell-green {
    background-color: #ccffcc;
  }
  .lemma-cell-red {
    background-color: #ff9999;
  }
  .lemma-cell-violet {
    background-color: #d0b0e0;
  }
  .final-theorem-green {
    background-color: #ccffcc;
  }
  .final-theorem-red {
    background-color: #ff9999;
  }
</style>
</head>
<body>
'''

def generate_problem_rows(problem, version, label):
    tac_num = num_of_tactics[version]
    problem_text = escape_html(problem.get('problem', '')).replace('&lt;br&gt;', '<br>')
    init_structured_solution = escape_html(problem.get('init_structured_solution', ''))
    structured_solution = escape_html(problem.get('structured_solution', ''))
    final_theorems = [escape_html(problem.get(f'final_theorem_{i}', '')) for i in range(tac_num)]
    final_error_messages = [escape_html(problem.get(f'error_{i}', '')) for i in range(tac_num)]
    final_results = [problem.get(f'final_result_{i}', False) for i in range(tac_num)]

    ft_color_classes = ['final-theorem-green' if final_result else 'final-theorem-red' for final_result in final_results]
    problem_ft_color = 'final-theorem-green' if any(final_results) else 'final-theorem-red'

    return f'''
    <tr class="problem-row">
      <td class="{problem_ft_color}" colspan="{5 + 2 * tac_num}">{label}</td>
    </tr>
    <tr class="problem-row">
      <td class="{problem_ft_color}">{problem_text}</td>
      <td colspan><div class="structured-solution">{init_structured_solution}</div></td>
      <td colspan=3><div class="structured-solution">{structured_solution}</div></td>''' + ''.join(f'''
      <td class="{ft_color_class}"><pre>{final_error_message}</pre></td>
      <td class="{ft_color_class}"><pre>{final_theorem}</pre></td>'''
    for ft_color_class, final_error_message, final_theorem in zip(ft_color_classes, final_error_messages, final_theorems)
    ) + '\n    </tr>\n'


def generate_lemma_row(lemma):
    unproved = escape_html(lemma.get('unproved', ''))
    proved = escape_html(lemma.get('proved', ''))
    error = escape_html(lemma.get('error', ''))
    name = lemma.get('name', '')

    if 'NO FILE' in proved:
        lemma_color = 'red'
    elif 'sorry' in proved:
        lemma_color = 'violet'
    else:
        lemma_color = 'green'

    return f'''
        <tr class="lemma-row">
          <td></td>
          <td></td>
          <td> {name} </td>
          <td class="lemma-cell-{lemma_color}"><pre>{unproved}</pre></td>
          <td class="lemma-cell-{lemma_color}"><pre>{proved}</pre></td>
          <td class="lemma-cell-{lemma_color}"><pre>{error}</pre></td>
          <td></td>
        </tr>
    '''

def generate_problem_record(problem, version, label):
    html = generate_problem_rows(problem, version, label)
    html += ''.join(generate_lemma_row(lemma) for lemma in problem.get('lemmas', []))
    return html

def generate_section(section, section_description, section_filter) :
    section_length = len([problem_index for problem_index in range(len(reports['old'])) if section_filter(problem_index)])
    table_width = max(num_of_tactics.values())
    html = f'''
      <h2>{section_description} : {section_length}</h2>
      <table>
      <tr>
        <th>Problem</th>
        <th>Initial Structured Solution</th>
        <th>Structured Solution</th>
        <th>Unproved</th>
        <th>Proved</th>''' + ''.join(f'''
        <th>Error {i}</th>
        <th>Final Theorem {i}</th>'''
    for i in range(table_width)
    ) + '\n      </tr>\n'
    
    for problem_index in range(len(reports['old'])): 
        # Generate and add problem row
        if section_filter(problem_index):
            for version in ['old', 'new']:
                html += generate_problem_record(reports[version][problem_index], version, version.upper() + f' (current section: {section_description} ) ')         

    html += f'''
      </table>
    '''
    return html

def generate_header():
    old_result = len(evaluation_info['old']['correct'])
    new_result = len(evaluation_info['new']['correct']) 
    return f"<h1> A/B Testing report. Old result: {old_result} New result: {new_result}</h1>\n"

#################### GENERATE SECTIONS #################################################

section_name = {
  'red_green' : "Theorems that were incorrect, but now are correct",
  'green_red' : "Theorems that were correct, but now are incorrect",
  'red_red' : "Theorems that were incorrect and remain incorrect",
  'green_green' : "Theorems that were correct and remain correct"
}

section_filter = {
  'red_green' : lambda x : x in evaluation_info['old']['incorrect'] and x in evaluation_info['new']['correct'],
  'green_red' : lambda x : x in evaluation_info['old']['correct'] and x in evaluation_info['new']['incorrect'],
  'red_red' : lambda x : x in evaluation_info['old']['incorrect'] and x in evaluation_info['new']['incorrect'],
  'green_green' : lambda x : x in evaluation_info['old']['correct'] and x in evaluation_info['new']['correct']
}

html += generate_header()

for section in section_name:
    html += generate_section(section, section_name[section], section_filter[section])

html += ''' </body>
    </html>
    '''

# Save the HTML to file
with open(output_file, 'w', encoding = 'utf-8') as f:
    f.write(html)

print("HTML report generated in ", output_file)
