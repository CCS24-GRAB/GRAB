from transformers import BertTokenizerFast
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser(description='Evaluation')
parser.add_argument('--model', default="bert-base-uncased", type=str, help='large language model')
parser.add_argument('--dataset', default="cola", type=str, help='dataset for task')
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--setting", default="benchmark", type=str, help="setting for evaluation")
parser.add_argument("--method", default="dlg", type=str, help="method for evaluation")
args = parser.parse_args()

model = args.model
setting = args.setting
batch_size = args.batch_size
dataset = args.dataset
method = args.method

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
R1s = []
R2s = []
RLs = []

for run in ["first", "second", "third"]:
    result_file = f"results/{setting}/{method}/{dataset}/{model}_{run}_run_{batch_size}_results.txt"
    with open(result_file, "r") as file:
        lines = file.readlines()

    model_out = []
    reference = []
    for i in range(len(lines)):
        if lines[i] == "Reference:\n":
            current_reference = []
            for j in range(batch_size):
                current_reference.append(
                    tokenizer.decode(tokenizer(lines[i + j + 1].strip())["input_ids"], skip_special_tokens=True))
            reference.append(current_reference)
        elif lines[i] == "Prediction:\n":
            current_solution = []
            for j in range(batch_size):
                current_solution.append(
                    tokenizer.decode(tokenizer(lines[i + j + 1].strip())["input_ids"], skip_special_tokens=True))
            model_out.append(current_solution)

    if len(model_out) > len(reference):
        model_out = model_out[:len(reference)]
    elif len(model_out) < len(reference):
        reference = reference[:len(model_out)]

    print("calculating rouge scores...")
    overall_scores = {"r1": [], "r2": [], "rl": []}
    rouge = ROUGEScore(accumulate="best")
    for i in range(len(model_out)):
        references = reference[i]
        solutions = model_out[i]
        batch_scores = {"r1": [], "r2": [], "rl": []}
        for ref in references:
            temp_scores = []
            for solution in solutions:
                temp_scores.append(rouge(preds=solution, target=ref))
            temp_r1 = []
            temp_r2 = []
            temp_rl = []
            for score in temp_scores:
                temp_r1.append(score["rouge1_fmeasure"].item())
                temp_r2.append(score["rouge2_fmeasure"].item())
                temp_rl.append(score["rougeL_fmeasure"].item())
            batch_scores["r1"].append(max(temp_r1))
            batch_scores["r2"].append(max(temp_r2))
            batch_scores["rl"].append(max(temp_rl))

        overall_scores["r1"].append(np.mean(batch_scores["r1"]))
        overall_scores["r2"].append(np.mean(batch_scores["r2"]))
        overall_scores["rl"].append(np.mean(batch_scores["rl"]))

    R1s.append(np.mean(overall_scores["r1"]))
    R2s.append(np.mean(overall_scores["r2"]))
    RLs.append(np.mean(overall_scores["rl"]))

# calculate the mean and variance
print(f"R1: {np.mean(R1s)} ± {np.std(R1s)}")
print(f"R2: {np.mean(R2s)} ± {np.std(R2s)}")
print(f"RL: {np.mean(RLs)} ± {np.std(RLs)}")
