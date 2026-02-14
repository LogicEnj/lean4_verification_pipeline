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
    yield -1
    tac_pattern = re.compile(r'^task-\d+-(\d+).lean$')
    for filename in os.listdir(directory):
        match = tac_pattern.fullmatch(filename)
        if match and os.path.isfile(os.path.join(directory, filename)):
            yield int(match[1])

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
        


################################################################################

csv_str_lemmas = get_input_file_or_folder('./config.yaml', 0)
dir_unproved = get_input_file_or_folder('./config.yaml', 1)
dir_proved = get_input_file_or_folder('./config.yaml', 2)
dir_final = get_input_file_or_folder('./config.yaml', 3)

output_file = get_output_file_or_folder('./config.yaml')

print('input:', csv_str_lemmas, dir_unproved, dir_proved, dir_final)
print('output:', output_file)

lemmas = pd.read_csv(csv_str_lemmas)
df = pd.read_csv(os.path.join(dir_proved, 'theorems.csv'))
print(len(df), "records")

num_of_tactics = max(tactics_num(dir_final)) + 1
lemma_inds = sorted(list(lemma_indices(dir_unproved)))
fact_inds = sorted(list(fact_indices(dir_unproved)))

data = []
for index, row in df.iterrows():
    task_number = index
    info = {}
    info['problem'] = row['problem'] + f'<br><br>\n Problem {task_number}'
    info['init_structured_solution'] = row['structured_solution']
    info['structured_solution'] = '\n'.join(_row_['lemma'] for _, _row_ in lemmas.iterrows() if _row_['task_number'] == task_number)
    for tactic_ind in range(num_of_tactics):
        final_path = os.path.join(dir_final, f"task-{task_number}-{tactic_ind}.lean") 
        info[f'final_theorem_{tactic_ind}'] = read_file_content(final_path)
        full_path_to_error_message_file = final_path[:-5] + '.err'
        info[f'error_{tactic_ind}'] = read_file_content(full_path_to_error_message_file)
        info[f'final_result_{tactic_ind}'] = info[f'error_{tactic_ind}'].endswith('TRUE')
    info['lemmas'] = [{'name': name, 'unproved': read_file_content(os.path.join(dir_unproved, filename)), 'proved': read_file_content(os.path.join(dir_proved, filename)), 'error': read_file_content(os.path.join(dir_proved, filename[:-5]+'.err'))}
                      for name, filename in facts_and_lemmas_for_task(task_number, lemma_inds, fact_inds)]
    data.append(copy.deepcopy(info))


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
<h2>Formal Verification Report</h2>
<table>
  <tr>
    <th>Problem</th>
    <th>Initial Structured Solution</th>
    <th>Structured Solution</th>
    <th>Unproved</th>
    <th>Proved</th>''' + ''.join(
            f'''
    <th>Error {i}</th>
    <th>Final Theorem {i}</th>'''
    for i in range(num_of_tactics)
    ) + '\n  </tr>\n'

for problem in data:
    problem_text = escape_html(problem.get('problem', '')).replace('&lt;br&gt;', '<br>')
    init_structured_solution = escape_html(problem.get('init_structured_solution', ''))
    structured_solution = escape_html(problem.get('structured_solution', ''))
    final_theorems = [escape_html(problem.get(f'final_theorem_{tactic_ind}', '')) for tactic_ind in range(num_of_tactics)]
    final_error_messages = [escape_html(problem.get(f'error_{tactic_ind}', '')) for tactic_ind in range(num_of_tactics)]
    final_results = [problem.get(f'final_result_{tactic_ind}', False) for tactic_ind in range(num_of_tactics)]

    ft_color_classes = ['final-theorem-green' if final_result else 'final-theorem-red' for final_result in final_results]
    problem_ft_color_class = 'final-theorem-green' if any(final_results) else 'final-theorem-red'

    # Add problem row
    html += f'''
    <tr class="problem-row">
      <td class="{problem_ft_color_class}">{problem_text}</td>
      <td colspan><div class="structured-solution">{init_structured_solution}</div></td>
      <td colspan=3><div class="structured-solution">{structured_solution}</div></td>''' + ''.join(f'''
      <td class="{ft_color_class}"><pre>{final_error_message}</pre></td>
      <td class="{ft_color_class}"><pre>{final_theorem}</pre></td>'''
    for ft_color_class, final_error_message, final_theorem in zip(ft_color_classes, final_error_messages, final_theorems)
    ) + '\n    </tr>\n'

    # Add lemma rows
    for lemma in problem.get('lemmas', []):
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

        html += f'''
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

html += '''
</table>
</body>
</html>
'''

# Save the HTML to file
with open(output_file, 'w', encoding = 'utf-8') as f:
    f.write(html)

print("HTML report generated in", output_file)
