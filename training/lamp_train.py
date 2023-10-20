import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, AdamW, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import matthews_corrcoef

np.random.seed(100)
torch.manual_seed(100)
device = 'cuda'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rotten_tomatoes'], default='cola')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--noise', type=float, default=None)
    parser.add_argument('--prune', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    args = parser.parse_args()

    seq_key = 'text' if args.dataset == 'rotten_tomatoes' else 'sentence'
    num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels,
                                                               attention_probs_dropout_prob=args.dropout,
                                                               hidden_dropout_prob=args.dropout).to(device)

    model.bert.embeddings.word_embeddings.weight.requires_grad = False
    model.bert.embeddings.position_embeddings.weight.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    tokenizer.model_max_length = 512
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.dataset == 'cola':
        test_mcc = load_metric('matthews_correlation')
        test_f1 = load_metric('f1')
        test_acc = load_metric('accuracy')
        train_mcc = load_metric('matthews_correlation')
        train_f1 = load_metric('f1')
        train_acc = load_metric('accuracy')
    else:
        metric = load_metric('accuracy')
        train_metric = load_metric('accuracy')

    def tokenize_function(examples):
        return tokenizer(examples[seq_key], truncation=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    if args.dataset in ['cola', 'sst2', 'rte']:
        datasets = load_dataset('glue', args.dataset)
    else:
        datasets = load_dataset(args.dataset)

    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    if args.dataset == 'cola' or args.dataset == 'sst2':
        tokenized_datasets = tokenized_datasets.remove_columns(['idx', 'sentence'])
    elif args.dataset == 'rotten_tomatoes':
        tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    else:
        assert False
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    opt = AdamW(model.parameters(), lr=5e-5)

    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    model.train()
    n_steps = 0
    train_loss = 0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            train_mcc.add_batch(predictions=predictions, references=batch['labels'])
            train_f1.add_batch(predictions=predictions, references=batch['labels'])
            train_acc.add_batch(predictions=predictions, references=batch['labels'])

            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()

            if args.noise is not None:
                for param in model.parameters():
                    if param.requires_grad:
                        grad_norm = param.grad.norm()
                        param.grad = (param.grad / grad_norm) + torch.randn(param.grad.shape).to(
                            device) * args.noise

            if args.prune is not None:
                for param in model.parameters():
                    if param.requires_grad:
                        param.grad = param.grad * (torch.rand(param.grad.shape).to(device) > args.prune).float()

            opt.step()
            lr_scheduler.step()
            opt.zero_grad()
            progress_bar.update(1)

            n_steps += 1
        print('epoch: ', epoch)
        train_mcc_score = train_mcc.compute()
        train_f1_score = train_f1.compute()
        train_acc_score = train_acc.compute()
        print('mcc train: ', train_mcc_score)
        print('f1 train: ', train_f1_score)
        print('acc train: ', train_acc_score)
        with open(f"../results/training/{args.dataset}_noise_{args.noise}_prune_{args.prune}_dropout_{args.dropout}.txt", "a") as file:
            file.write(f"epoch: {epoch}\n")
            file.write(f"mcc train: {train_mcc_score}\n")
            file.write(f"f1 train: {train_f1_score}\n")
            file.write(f"acc train: {train_acc_score}\n")

        train_loss = 0.0

        model.eval()
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            test_mcc.add_batch(predictions=predictions, references=batch['labels'])
            test_f1.add_batch(predictions=predictions, references=batch['labels'])
            test_acc.add_batch(predictions=predictions, references=batch['labels'])
        test_mcc_score = test_mcc.compute()
        test_f1_score = test_f1.compute()
        test_acc_score = test_acc.compute()
        print('mcc test: ', test_mcc_score)
        print('f1 test: ', test_f1_score)
        print('acc test: ', test_acc_score)
        with open(
                f"../results/training/{args.dataset}_noise_{args.noise}_prune_{args.prune}_dropout_{args.dropout}.txt",
                "a") as file:
            file.write(f"mcc test: {test_mcc_score}\n")
            file.write(f"f1 test: {test_f1_score}\n")
            file.write(f"acc test: {test_acc_score}\n")
            file.write("\n")
    # model.save_pretrained(f'../models/lamp_epoch_{args.num_epochs}_frozen')


if __name__ == '__main__':
    main()
