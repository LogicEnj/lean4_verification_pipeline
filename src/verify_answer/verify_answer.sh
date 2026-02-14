rm server*.log
pkill -f python

DATASET=$1
TIME="`date +%Y%m%d%H%M%S`"

# 0. Generate config for this dataset

python ../common/generate_config.py --dataset $DATASET --time $TIME

# 1. Convert solutions and answers into theorems:

../../scripts/run_formalizer.sh > server1.log 2>&1 &
SERVER_PID=$!
../../scripts/wait_for_llm.sh $SERVER_PID server1.log
python ../common/formalize.py
kill -9 $SERVER_PID
pkill -f "python"
gpustat
sleep 3
gpustat
sleep 3
gpustat
 
# 2. Calculate structured solutions:

../../scripts/run_qwen.sh > server2.log 2>&1 &
SERVER_PID=$!
../../scripts/wait_for_llm.sh $SERVER_PID server2.log
python qwen-structured-or-not.py
kill -9 $SERVER_PID
pkill -f "python"
gpustat
sleep 3
gpustat
sleep 3
gpustat

# 3. Construct proofs for theorems

../../scripts/run_prover.sh > server3.log 2>&1 &
SERVER_PID=$!
../../scripts/wait_for_llm.sh $SERVER_PID server3.log
python kimina_prover.py
kill -9 $SERVER_PID
pkill -f "python"
sleep 0.3

python verify_proofs.py




