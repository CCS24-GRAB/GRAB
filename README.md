# GRAB: Uncovering Gradient Inversion Risks in Practical Language Model Training

![image](Overview.png)

<h2> Environment setup </h2>
<h3>Install the conda environment with the following commands one by one:</h3>
<br>
<code>conda create -n GRAB python=3.9.4</code>
<br>
<code>conda activate GRAB</code>
<br>
<code>pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118</code>
<br>
<code>pip install transformers==4.34.1 joblib==1.3.2</code>

<h2> Experiments </h2>

<h3> Benchmark Settings (Figure 2 in paper)</h3>
From the root directory
<br>
<code>cd main_attack</code>

Create symbolic link of utils.
<br>
<code>ln -s ../utils utils</code>

<h4> Parameters Explanation</h4>
--device: the device to run experiments on, e.g. cuda:0
<br>
--model: the model to be attacked. Only use bert-base-uncased.
<br>
--dataset: the dataset for experiments. Only use cola, sst-2, or rotten_tomatoes.
<br>
--batch_size: the batch size for experiments. Choose from 1 to 32.
<br>
--lr_decay_type: the type of learning rate decay. Only use StepLR in this one.
<br>
--parallel: whether to use parallel computing in discrete optimization. If not specified, it will not use parallel computing.
<br>
--recover_batch: to recover from a specific batch, if experiment failed for some reason.
<br>
--run: the number of runs for each experiment. 

<code>python bert_easy_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE --parallel</code>

The results will be saved in <code>results/easy/DATASET</code> from root directory. Make sure you create these folders
before running the experiments.

#

<h3> Practical Settings (Figure 3 in paper) </h3>

From the root directory
<br>
<code>cd main_attack</code>

Create symbolic link of utils.
<br>
<code>ln -s ../utils utils</code>

<h4> Parameters Explanation</h4>
The same as the benchmarking experiment.

#

The below command will run our attack with dropout mask learning.

<code>python bert_hard_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE --parallel</code>

The results will be saved in <code>results/hard/DATASET</code> from root directory. Make sure you create these folders
before running the experiments.

#

The below command will run our attack without dropout mask learning.

<code>python bert_hard_no_DL_attack.py --device DEVICE --dataset DATASET --batch_size BATCH_SIZE --parallel</code>

The results will be saved in <code>results/hard_no_DL/DATASET</code> from root directory. Make sure you create these
folders
before running the experiments.