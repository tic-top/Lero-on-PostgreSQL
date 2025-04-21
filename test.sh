# for lero model stats

# # tpch, tpcds, job(imdb), stats
# ModelPath = ./reproduce/stats_pw

# python server.py

# # DB = stats in conf.py

# python test.py --query_path ../reproduce/test/stats.txt --output_query_latency_file stats.test

# # https://github.com/gregrahn/join-order-benchmark

# ## visulaize stats.test

#!/bin/bash

# Define the models and their corresponding test files
declare -A models=(
    ["stats_pw"]="stats.txt"
    ["imdb_pw"]="imdb.txt"
    ["tpch_pw"]="tpch.txt"
)

# Backup the original server.conf
cp server.conf server.conf.bak

# Function to test a model
test_model() {
    local model=$1
    local test_file=$2
    
    echo "Testing model: $model"
    
    # Update server.conf with the current model path
    sed -i "s|^ModelPath = .*|ModelPath = ./reproduce/$model|" server.conf
    
    # Update conf.py with the current DB
    local db=$(echo $model | cut -d'_' -f1)
    sed -i "s/^DB = .*/DB = $db/" conf.py
    
    # Start the server in background and get its PID
    python server.py &
    SERVER_PID=$!
    
    # Wait for server to start (adjust sleep time if needed)
    sleep 5
    
    # Run the test
    python test.py --query_path "../reproduce/test_query/$test_file" --output_query_latency_file "$db.test"
    
    # Kill the server
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    echo "Finished testing $model"
    echo "----------------------------------------"
}

# Test each model
for model in "${!models[@]}"; do
    test_file="${models[$model]}"
    test_model "$model" "$test_file"
done

# Restore the original server.conf
mv server.conf.bak server.conf

echo "All tests completed!"