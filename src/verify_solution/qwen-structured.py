from custom_tools.config_reader import get_input_file_or_folder, get_output_file_or_folder 
import pandas as pd
import json
import os
from qwen_shared import tokenizer, get_qwen_answer, filter_answer

prompt_without_introduced = r"""You are a math expert. Please solve the following problem by reasoning step by step 
and then summarize your solution in propositional logic so that it 
satisfies the following criteria: The solution should be split into several easy to understand steps, 
each written in propositional logic as A_1 ∧ ... ∧ A_n -> B as a set of premises (explicitly stated, not as A_i) and one conclusion. 
You should include all relevant premises so that the proposition is true without previous context. Do not write text comments in the solution. Your summary should be in latex code.
Once you have obtained the necessary steps, stop thinking mode and put your summary between 
### Summary in Propositional Logic
and
### End of Summary
headers.
I will give you examples of such summaries:

begin of example section
Problem: Evaluate $(1+2i)6-3i$.
Solution: Distribute the factor of 6 and simplify to obtain $(1+2i)6-3i=6+12i-3i=\boxed{6+9i}$.
Summary:
1. $ (1 + 2i)6 = 6 + 12i $ -> $ (1 + 2i)6 - 3i = 6 + 12i - 3i $
2. $ (1 + 2i)6 - 3i = 6 + 12i - 3i $ -> $ (1 + 2i)6 - 3i = 6 + 9i $

Problem: If $f(x) = \frac{3x - 2}{x - 2}$, what is the value of $f(-2) + f(-1) + f(0)$? Express your answer as a common fraction.
Solution: $f(-2) + f(-1) + f(0) = \frac{3(-2)-2}{-2-2} + \frac{3(-1)-2}{-1-2} + \frac{3(0)-2}{0-2} = \frac{-8}{-4} + \frac{-5}{-3} + \frac{-2}{-2} = 2 + \frac{5}{3} + 1 = \boxed{\frac{14}{3}}$
Summary:
1. $ f(x) = \frac{3x - 2}{x - 2}$ -> $ f(-2) = 2 $
2. $ f(x) = \frac{3x - 2}{x - 2}$ -> $f(-1) = \frac{5}{3} $
3. $ f(x) = \frac{3x - 2}{x - 2}$ -> $ f(0) = 1 $
4. $ f(-2) = 2 \land f(-1) = \frac{5}{3} \land f(0) = 1 $  -> $ f(-2) + f(-1) + f(0) = \frac{14}{3} $

Problem: What integer $x$ satisfies $\frac{1}{4}<\frac{x}{7}<\frac{1}{3}$?
Solution: Multiplying all expressions in the inequalities by $7$, we have $\frac{7}{4} < x < \frac{7}{3}$. Since $\frac{7}{4}$ is between $1$ and $2$, and $\frac{7}{3}$ is between $2$ and $3$, the only integer $x$ between these two fractions is $\boxed{2}$.
Summary:
1. $ \frac{1}{4} < \frac{x}{7} < \frac{1}{3} $ -> $ \frac{7}{4} < x < \frac{7}{3} $
2. $ \frac{7}{4} < x $ -> $ x \geq \lceil \frac{7}{4} \rceil $
3. $ x \geq \lceil \frac{7}{4} \rceil $ -> $x \geq 2 $
4. $ x < \frac{7}{3} $ -> $ x \leq \lfloor \frac{7}{3} \rfloor $
5. $ x \leq \lfloor \frac{7}{3} \rfloor $ -> $x \leq 2$
6. $ (x \geq 2) \land (x \leq 2) $ -> $ x = 2 $

Problem: If $10^x - 10 = 9990$, what is $x$ equal to?
Solution: Since $10^x - 10 = 9990$, we have $10^x = 9990+10=10000$. $If $10^x = 10000$, then $x=\boxed{4}$, since $10000$ ends in four zeroes.
Summary: 
1. $ 10^x - 10 = 9990 $ -> $ 10^x = 10000 $
2. $ (10^x = 10000) \land (10000 = 10^4) $ -> $ 10^x = 10^4 $
3. $ 10^x = 10^4 $ -> $ x = 4 $.
end of example section
"""

prompt_with_introduced = r"""You are a math expert. Please solve the following problem by using the given variables and reasoning step by step 
and then summarize your solution in propositional logic so that it 
satisfies the following criteria: The solution should be split into several easy to understand steps, 
each written in propositional logic as A_1 ∧ ... ∧ A_n -> B as a set of premises (explicitly stated, not as A_i) and one conclusion. 
You should include all relevant premises so that the proposition is true without previous context. Your summary should be in latex code. 
Once you have obtained the necessary steps, stop thinking mode and put your summary between 
### Summary in Propositional Logic
and
### End of Summary
headers.
I will give you examples of such summaries:

begin of example section
Problem: Find the product of $6_8 \cdot 7_8$. Express your answer in base $8$.
Variables:
$x$ is $6_8$
$y$ is $7_8$
Solution: We see that $x = 6_{10}$ and $y = 7_{10}$ in decimal. Multiplying them, we have $x \cdot y = 42_{10} = 52_8$. Thus, the answer is $\boxed{52_8}.$
Summary:
1. $ (x = 6_8) \land (6_8 = 6_{10}) $ -> $ x = 6_{10} $
2. $ (y = 7_8) \land (7_8 = 7_{10}) $ -> $ y = 7_{10} $
3. $ (x = 6_{10}) \land (y = 7_{10}) $ -> $ x \cdot y = 42_{10} $
4. $ (x \cdot y = 42_{10}) \land (42_{10} = 52_8) $ -> $ x \cdot y = 52_8 $

Problem: Siddhartha Gautama has 17 coins, all uranium and plutonium. Uranium and plutonium coin costs 4 and 7 rupees, respectively. In total, the coins are worth 74 rupees. How many uranium coins does he have?
Variables:
u is the number of uranium coins
p is the number of plutonium coins
Solution: We know $ u + p = 17 $ and $ 4u + 7p = 74 $. Express $p$ as $ p = 17 - u$ and substitute into \( 4u + 7p = 74 \), we obtain $ 4u + 7(17 - u) = 74 $. After expanding braces and simplifying the left side we have $ 119 - 3u = 74 $. After taking $3u$ and $74$ to the opposite sides the equation becomes $ 45 = 3u $ with the solution $u = 15$. Thus, the answer is $15$.
Summary:
1. $ u + p = 17 $ -> $ p = 17 - u $
2. $ (p = 17 - u) \land (4u + 7p = 74) $ -> $ 4u + 7(17 - u) = 74 $
3. $ 4u + 7(17 - u) = 74 $ -> $ 119 - 7u + 4u = 74 $
4. $ 119 - 7u + 4u = 74 $ -> $ 119 - 3u = 74 $
5. $ 119 - 3u = 74 $ -> $ 45 = 3u $
6. $ 45 = 3u $ -> $ u = 15 $

Problem: What power of 27 is equal to 81?
Variables:
$x$ is the exponent such that $27^x = 81$
Solution: We are asked to solve $27^x=81$ for $x$.  Writing $27$ as $3^3$ and $81$ as $3^4$, the equation becomes $(3^3)^x=3^4$.  The left-hand side simplifies to $3^{3x}$, so we may set exponents equal to find $3x=4$, which implies $x=\\boxed{\\frac{4}{3}}$.
Summary:
1. $ (27^x = 81) \land (27 = 3^3) \land (81 = 3^4) $ -> $ (3^3)^x = 3^4 $
2. $ ((3^3)^x = 3^(3x)) \land (3^3)^x = 3^4 $ -> $ 3^(3x) = 3^4 $
3. $ 3^(3x) = 3^4 $ -> $ 3x = 4 $
4. $ 3x = 4 $ -> $ x = \frac{4}{3} $
end of example section

Put your summary between 
### Summary in Propositional Logic
and
### End of Summary
headers.

"""

def get_text_prompt(row_modified):
    if row_modified['was_variables'] or not row_modified['variables']:
        prompt = prompt_without_introduced
        add_info = ''
    else:
        prompt = prompt_with_introduced
        add_info = 'Variables:\n' + '\n'.join(row_modified['definitions']) + '\n'
    prompt += f"Problem:\n{row_modified['problem']}\n" + add_info
    return prompt

##################################################################
input_file = get_input_file_or_folder('./config.yaml')
output_file = get_output_file_or_folder('./config.yaml')
log_dir = "./logs/structured"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

print('input:', input_file)
print('output:', output_file)

batch_size = 10

with open(input_file, 'r', encoding = 'utf-8') as file:
    data = [json.loads(line) for line in file.readlines()]

with open(os.path.join(log_dir, "prompt.txt"), 'w') as file:
    file.write(get_text_prompt(data[0]))

df = pd.DataFrame(data)
raw_output = get_qwen_answer(df, batch_size, lambda row: row, tokenizer, get_text_prompt)
df['structured_solution'] = filter_answer(raw_output)
df['raw_output'] = raw_output
df.to_json(output_file, lines = True, orient = 'records')
