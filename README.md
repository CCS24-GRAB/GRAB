# GRAB: A Practical Gradient Inversion with Hybrid Optimization on Language Models

![image](Overview.png)

<h2> Environment setup </h2>
<h3>Install the conda environment with the following commands one by one:</h3>
<br>
<code>conda create -n GRAB python=3.9.4</code>
<br>
<code>pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118</code>
<br>
<code>pip install transformers joblib</code>

<h2> Experiments </h2>

<h3> Benchmarking (Table 3 in paper)</h3>
From the root directory
<br>
<code>cd main_attack</code>

Create symbolic link of utils.
<br>
<code>ln -s ../utils utils</code>

<h4> Parameters Explanation</h4>
--device: the device to run experiments on, e.g. cuda:0
<br>
--model: the model to be attacked. Only use bert-base-uncased in this one.
<br>
--dataset: the dataset for experiments. Only use cola, sst-2, or rotten_tomatoes in this one.
<br>
--batch_size: the batch size for experiments. For this experiment, only choose between 1, 2, and 4.
<br>
--lr_decay_type: the type of learning rate decay. Choose between only use StepLR in this one.
<br>
--parallel: whether to use parallel computing in discrete optimization. If not specified, it will not use parallel computing.
<br>
--recover_batch: to recover from a specific batch, if experiment failed for some reason.
<br>
--run: the number of runs for each experiment. For this experiment, only choose between 1, 2, and 3.

<code>python bert_easy_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE --parallel</code>

The results will be saved in <code>results/easy/DATASET</code> from root directory. Make sure you create these folders
before running the experiments.

<h3> Larger Batch Sizes (Table 4 in paper)</h3>
From the root directory
<br>
<code>cd main_attack</code>

Create symbolic link of utils.
<br>
<code>ln -s ../utils utils</code>

<h4> Parameters Explanation</h4>
--batch_size: the batch size for experiments. For this experiment, only choose between 8, 16, and 32.
<br>
All the rest are the same as the benchmarking experiment.

<code>python bert_easy_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE</code>

The results will be saved in <code>results/easy/DATASET</code> from root directory. Make sure you create these folders
before running the experiments.

<h3> Practical Settings (Table 5 in paper) </h3>

From the root directory
<br>
<code>cd main_attack</code>

Create symbolic link of utils.
<br>
<code>ln -s ../utils utils</code>

<h4> Parameters Explanation</h4>
The same as the benchmarking experiment.

<code>python bert_hard_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE --parallel</code>

The results will be saved in <code>results/hard/DATASET</code> from root directory. Make sure you create these folders
before running the experiments.

<code>python bert_hard_no_DL_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE --parallel</code>

The results will be saved in <code>results/hard_no_DL/DATASET</code> from root directory. Make sure you create these
folders
before running the experiments.