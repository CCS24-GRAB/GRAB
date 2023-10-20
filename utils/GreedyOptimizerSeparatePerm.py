import torch
import joblib
import copy
import random
from torch.nn.functional import one_hot
import numpy as np


class GreedyOptimizerSeparatePerm:
    def __init__(self, client, server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                 labels, longest_length, separate_tokens, num_perms=1000, alpha=0.01,
                 continuous_solution=None, discrete_solution=None):
        self.client = client
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.client_grads = client_grads
        self.result_file = result_file
        self.solution = None
        self.obj_value = None
        self.labels = labels
        self.longest_length = longest_length
        self.separate_tokens = separate_tokens
        self.continuous_solution = continuous_solution
        self.discrete_solution = discrete_solution
        self.num_perms = num_perms
        self.alpha = alpha
        self.server.eval()

    def set_up(self):
        perms = []
        objs = []
        perms.append(self.continuous_solution)
        objs.append(self.calculate_obj_value(self.continuous_solution))
        if self.discrete_solution is not None:
            perms.append(self.discrete_solution)
            objs.append(self.calculate_obj_value(self.discrete_solution))

        real_tokens = torch.LongTensor(self.continuous_solution)[:, 1:self.longest_length - 1]
        for i in range(self.num_perms):
            if i % 100 == 0:
                print(f"Evaluating continuous perm {i}")
            idx = torch.randperm(self.longest_length - 2)
            perm = real_tokens[:, idx]
            perm = perm.tolist()
            for j in range(self.batch_size):
                sequence = perm[j]
                sequence = [self.tokenizer.cls_token_id] + sequence + [self.tokenizer.sep_token_id]
                sequence = sequence + [self.tokenizer.pad_token_id] * (self.seq_len - len(sequence))
                perm[j] = sequence
            perms.append(perm)
            objs.append(self.calculate_obj_value(perm))

        if self.discrete_solution is not None:
            real_tokens = torch.LongTensor(self.discrete_solution)[:, 1:self.longest_length - 1]
            for i in range(self.num_perms):
                if i % 100 == 0:
                    print(f"Evaluating discrete perm {i}")
                idx = torch.randperm(self.longest_length - 2)
                perm = real_tokens[:, idx]
                perm = perm.tolist()
                for j in range(self.batch_size):
                    sequence = perm[j]
                    sequence = [self.tokenizer.cls_token_id] + sequence + [self.tokenizer.sep_token_id]
                    sequence = sequence + [self.tokenizer.pad_token_id] * (self.seq_len - len(sequence))
                    perm[j] = sequence
                perms.append(perm)
                objs.append(self.calculate_obj_value(perm))

        self.solution = perms[objs.index(max(objs))]
        self.obj_value = max(objs)

    def calculate_obj_value(self, solution):
        dummy_x = torch.tensor(solution).to(self.device)
        dummy_y = torch.LongTensor(self.labels).to(self.device)
        dummy_attention_mask = []
        for i in range(len(solution)):
            mask = []
            for j in range(len(solution[i])):
                if solution[i][j] == self.tokenizer.pad_token_id:
                    mask.append(0)
                else:
                    mask.append(1)
            dummy_attention_mask.append(mask)
        dummy_outputs = self.server(input_ids=dummy_x,
                                    attention_mask=torch.tensor(dummy_attention_mask).to(self.device))
        dummy_preds = dummy_outputs.logits
        loss_function = torch.nn.CrossEntropyLoss()
        dummy_loss = loss_function(dummy_preds, dummy_y)
        server_parameters = []
        for param in self.server.parameters():
            if param.requires_grad:
                server_parameters.append(param)
        server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
        grads_diff = 0
        # for gx, gy in zip(server_grads, self.client_grads):
        #     grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
        # Calculate the cosine distance between the gradients of the server and the client
        for gx, gy in zip(server_grads, self.client_grads):
            grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
            # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))
        # Add the negative value such that bigger objective value means a better solution
        return -grads_diff.item()

    def optimize(self):
        self.set_up()
        current_best_solution = copy.deepcopy(self.solution)
        current_best_obj = self.obj_value
        global_best_solution = copy.deepcopy(self.solution)
        global_best_obj = self.obj_value
        i = 0
        while True:
            for j in range(self.batch_size):
                separate_tokens = self.separate_tokens[j]
                for k in range(1, self.longest_length - 1):
                    for token in [token for token in separate_tokens if token != self.tokenizer.cls_token_id]:
                        sequence = copy.deepcopy(current_best_solution[j])
                        sequence[k] = token
                        temp_solution = copy.deepcopy(current_best_solution)
                        temp_solution[j] = sequence
                        temp_obj = self.calculate_obj_value(temp_solution)
                        if temp_obj > current_best_obj:
                            current_best_obj = temp_obj
                            current_best_solution = copy.deepcopy(temp_solution)
            for j in range(self.batch_size):
                print(self.tokenizer.decode(current_best_solution[j]))
            if current_best_obj > global_best_obj:
                global_best_obj = current_best_obj
                global_best_solution = copy.deepcopy(current_best_solution)
            else:
                self.solution = global_best_solution
                self.obj_value = global_best_obj
                break
            print("Iteration: {}, Objective Value: {}".format(i, current_best_obj))
            with open(self.result_file, "a") as file:
                file.write(f"iter {i}:\n")
                for i in range(len(global_best_solution)):
                    file.write(f"{self.tokenizer.decode(global_best_solution[i], skip_special_tokens=True)}\n")
                file.write(f"{current_best_obj}\n")
            i += 1
        return self.solution, self.obj_value
