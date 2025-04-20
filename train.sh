## lero
python train_model.py --query_path tpch_train.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch.log --model_prefix tpch_test_model --topK 3

## tcnn

## tgnn

## ctnn


## postgres
python train_model.py --query_path tpch_train.txt  --algo pg --output_query_latency_file pg_tpch_train.log
python train_model.py --query_path tpch_test.txt --algo pg --output_query_latency_file pg_tpch_test.log