rm server*.log
pkill -f python

DATASET=$1

#$2 is an integer number; 0 - introduce variables by qwen; other numbers - do not introduce

if [ -z $3 ]; then
TIME="`date +%Y%m%d%H%M%S`"
else
TIME=$3
fi

# 0. Generate config for this dataset
python ../common/generate_config.py --dataset $DATASET --time $TIME

../../scripts/run_qwen.sh > server1.log 2>&1 &
SERVER_PID=$!
../../scripts/wait_for_llm.sh $SERVER_PID server1.log

# 1. Get variables with definitions from theorems:
python get_variables_for_problems.py --introduce $2
# 2. Compute structured solutions:
python qwen-structured.py
# 3. Lemmas of structured solutions to csv
python create_lemmas_csv.py
# 4. Full independent descriptions of every problem and lemma
python var_descriptions_for_every_statement.py

kill -9 $SERVER_PID
pkill -f "python"
gpustat
sleep 3
gpustat
sleep 3
gpustat

# 5. Correct descriptions of lemmas with context of problem and previous lemmas
python var_descriptions.py

../../scripts/run_formalizer.sh > server2.log 2>&1 &
SERVER_PID=$!
../../scripts/wait_for_llm.sh $SERVER_PID server2.log

# 6. Formalize theorems
python formalize.py
# 7. Formalize lemmas
python lemmas_from_structured.py

kill -9 $SERVER_PID
pkill -f "python"
gpustat
sleep 3
gpustat
sleep 3
gpustat

# 8. Fix some errors in theorem statements
python correct_theorems.py
# 9. Fix some errors in statements of unproved lemmas
python correct_unproved_lemmas.py

# 10. Check correctness of problem formalizations
#Stop if all formalizations are incorrect
res=$(python check_correctness.py)
if [[ $res == "all_formalizations_are_incorrect" ]]; then
exit 0
fi
 
../../scripts/run_prover.sh > server3.log 2>&1 &
SERVER_PID=$!
../../scripts/wait_for_llm.sh $SERVER_PID server3.log

# 11. Prove formalized lemmas
python kimina_prover_for_lemmas.py

kill -9 $SERVER_PID
pkill -f "python"
gpustat
sleep 3
gpustat
sleep 3
gpustat

# 12. Assemble lemmas into potential proofs of the theorem
python assemble_proofs_from_lemmas.py
# 13. Check if at least one of tentative proofs is valid
python verify_multiple_proofs.py
# 14. Create a report by lemmas
python report_by_lemmas.py
