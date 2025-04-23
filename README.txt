THIS IS THE README FOR THE PROJECT SUBMISSION.

As per syllabus:

################################ PART 1 ################################################################

"command line arguments and pointers to some small text data that can be used to run your tool"

This part is found at the end of Install.txt, but I also pasted it here for your REFERENCES
To run training:

set "port" in lero/test_script/config.py from 54321 to 5432 (default postgres port)
And make sure to set your username and password. Generally, a lot of bugs can be 
solved via fixing these files:

lero/test_script/config.py

lero/server.conf

Now to run, in one terminal/command line tab in "lero/":

    python server.py

Make sure this server is always running, and that the patched postgres is always running, to execute the parts below.

In another tab, in "lero/test_script":
    python train_model.py --query_path tpch_train.txt --test_query_path tpch_test.txt --algo lero --query_num_per_chunk 20 --output_query_latency_file lero_tpch.log --model_prefix tpch_test_model --topK 3

This should begin training lero and generating logs. Note that modifications to the entire codebase
since our last successful run may cause difficulty or inaccuracy in outputs, but in general it should still run.
Four kinds of files will be generated gradually during the execution of this script:

    lero_tpch.log
    The best plan considered by model in Lero will be executed and the results will be output to this file.
    lero_tpch.log_exploratory
    Other plans for pairwise training of each query will be executed and output to this file.
    lero_tpch.log.training
    Integrate the results of "lero_tpch.log" and "lero_tpch.log_exploratory" for model training.
    lero_tpch.log_tpch_test_model_i
    The performance of the model after i-th training.


In order to compare the results, after Lero executes all the queries, we use PostgreSQL to execute them again. 
The plans of the training set and the test set will be saved in "pg_tpch_train.log" and "pg_tpch_test.log" respectively:

    python train_model.py --query_path tpch_train.txt  --algo pg --output_query_latency_file pg_tpch_train.log
    python train_model.py --query_path tpch_test.txt --algo pg --output_query_latency_file pg_tpch_test.log

Then, run visualization.ipynb to generate train and test images. Double check which log files it is reading from,
as it may be looking for ttnn log files rather than lero ones. Check these lines:

    In Cell 3: lero_run_time_dict = convert_runtime_file_to_dict("ttnn_tpch.log")
    In Cell 5: lero_run_time_test.append(sum_runtime_file("ttnn_tpch.log_ttnn_tpch_test_model_" + str(i)))

When running for Lero, change these to:

    In Cell 3: lero_run_time_dict = convert_runtime_file_to_dict("lero_tpch.log")
    In Cell 5: lero_run_time_test.append(sum_runtime_file("lero_tpch.log_tpch_test_model_" + str(i)))


TO TRAIN OUR TTNN MODEL, simply run:
    ./train.sh

you may have to run this first:
    chmod +x train.sh

To visualize our model output, revert visualizatio back to:

    In Cell 3: lero_run_time_dict = convert_runtime_file_to_dict("ttnn_tpch.log")
    In Cell 5: lero_run_time_test.append(sum_runtime_file("ttnn_tpch.log_ttnn_tpch_test_model_" + str(i)))

Then run.


################################ PART 2 ################################################################

The modification (Our transformer), and the main functionality is defined in:
    /lero/TreeConvolution/ttnn.py   (new file created by us)

Additionally some updates were made to:
    lero/model.py
    lero/server.py
    lero/test_script/train_model.py
    lero/train.py
    lero/test_script/visualization.ipynb

But these are mostly for integration. These updated files already existed
for Lero infrastructure, and we updated them to integrate our ttnn.py