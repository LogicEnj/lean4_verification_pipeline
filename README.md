# Installation

1. Install Lean according to [this manual](https://docs.lean-lang.org/lean4/doc/setup.html)

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf --ssl-no-revoke| sh
```

2. Build lean_project

```bash
cd lean_project
lake build
```

3. Set PYTHONPATH variable to point at the src folder of the project (or add it to .profile or your python evironnent),
for example

```bash
cd src
pwd
```

```output
/mnt/compression/Verification/src
```

```bash
printf "\n\nexport PYTHONPATH=$PYTHONPATH:/mnt/compression/Verification/src\n" >> ~/.profile
source ~/.profile
```

4. Test lean client

```bash
cd src/test
python test_lean_client.py
```

5. Install vllm 0.10.1.1, pandas
```bash
pip install vllm==0.10.1.1
pip install pandas
```

6. Set absolute paths to Qwen, Kimina-Prover, Kimina-Autoformalizer in src/custom_tools/model_tokenizer.py, scripts/run_formalizer.sh, scripts/run_prover.sh, scripts/run_qwen.sh

The pipeline is configured to work with Qwen3-8B, Kimina-Autoformalizer-7B, Kimina-Prover-Preview-Distill-7B, and [this Mathlib version](https://github.com/leanprover-community/mathlib4/tree/ecd33ba785aacf04b696fd0763a8f51e27197ec9) that was copied in /lean_project/mathlib-old

# Verifying answers

1. Go to project folder

```bash
cd src/verify_answer
```

2. Select dataset from {easy, similar, Math-500} (or add your own dataset with name \<name\> with path data/datasets/Math/\<name\>.json) to generate formal proofs for problems in this dataset:

```bash
./verify_answer.sh easy
```

3. If you want verify formal proofs separately:

./verify_proofs.sh

You can find the intermediate results in data/calculation/verify_answer/\<time\>. Final results in data/result/verify_answer/\<time\>/\<dataset\>. (\<time\> is the time of starting the pipeline, \<dataset\> is the dataset name)

# Verifying solutions

1. Go to project folder

```bash
cd src/verify_solution
```

2. Select dataset from {easy, similar, Math-500} and run verification pipeline for this dataset (or add your own dataset with name \<name\> with path data/datasets/Math/\<name\>.json):

```bash
./verify_solution.sh similar 1
```
There similar is the dataset name, 1 indicates that qwen should introduce variables to solve the problem (when 0 indicates that qwen should not introduce variables to solve the problem)

You can find the intermediate results in data/calculation/\<time\>. Final results in data/result/verify_solution/\<time\>/\<dataset\>. (\<time\> is the time of starting the pipeline, \<dataset\> is the dataset name)

# A/B Testing of solution verification pipeline

1. Go to project folder

```bash
cd src/verify_solution
```

2. Select first branch and a dataset from {easy, similar, Math-500} and run verification pipeline (argument for verify_solution.sh are described in "Verifying solutions" section) for this dataset (or add your own dataset with name \<name\> with path data/datasets/Math/\<name\>.json):

```bash
git checkout branch_A
./verify_solution.sh similar 1
```

3. Save config file for creating A/B report:

```bash
cp config.yaml config_old.yaml
```

4. Run the pipeline for the second branch:

```bash
git checkout branch_B
./verify_solution.sh similar 1
```

5. Create A/B report, also known as 'report on change':

```bash
python report_on_change.py
```
