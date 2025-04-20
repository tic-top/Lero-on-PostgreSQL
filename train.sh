#!/usr/bin/env bash

model_name='lero'
dataset_names=(tpch tpcds job) # stats

for dataset_name in "${dataset_names[@]}"; do
  echo "===  $dataset_name ==="

  python train_model.py \
    --query_path "${dataset_name}_train.txt" \
    --test_query_path "${dataset_name}_test.txt" \
    --algo "${model_name}" \
    --query_num_per_chunk 20 \
    --output_query_latency_file "${model_name}_${dataset_name}.log" \
    --model_prefix "${model_name}_${dataset_name}_test_model" \
    --topK 3

  python train_model.py \
    --query_path "${dataset_name}_train.txt" \
    --algo pg \
    --output_query_latency_file "pg_${dataset_name}_train.log"

  python train_model.py \
    --query_path "${dataset_name}_test.txt" \
    --algo pg \
    --output_query_latency_file "pg_${dataset_name}_test.log"

  echo
done
