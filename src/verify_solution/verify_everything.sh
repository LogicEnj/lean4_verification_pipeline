#!/usr/bin/env bash

DATASETS=("easy" "similar" "Math-500")  # Elements contain spaces

TIME="`date +%Y%m%d%H%M%S`"
# Correct (uses [@])
for dataset in "${DATASETS[@]}"; do
    echo "Processing: $dataset"
    ./verify_solution.sh $dataset $1 $TIME
done
