# for lero model stats

# tpch, tpcds, job(imdb), stats
ModelPath = ./reproduce/stats_pw

python server.py

# DB = stats in conf.py

python test.py --query_path ../reproduce/test/stats.txt --output_query_latency_file stats.test

# https://github.com/gregrahn/join-order-benchmark

## visulaize stats.test