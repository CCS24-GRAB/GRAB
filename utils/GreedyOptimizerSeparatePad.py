import torch
import joblib
import copy
import random
from torch.nn.functional import one_hot
import numpy as np


class GreedyOptimizerSeparatePad():
    def __init__(self, client, server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                 labels, longest_length, separate_tokens, alpha=0.5, num_of_iter=5, prev_solution=None):
        self.client = client
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.client_grads = client_grads
        self.num_of_iter = num_of_iter
        self.result_file = result_file
        self.solution = None
        self.obj_value = None
        self.labels = labels
        self.longest_length = longest_length
        self.separate_tokens = separate_tokens
        self.alpha = alpha
        self.server.eval()

    def set_up(self):
        self.solution = []
        sequence = [self.tokenizer.cls_token_id]
        for i in range(self.longest_length - 2):
            sequence.append(self.tokenizer.pad_token_id)
        sequence.append(self.tokenizer.sep_token_id)
        sequence = sequence + [self.tokenizer.pad_token_id] * (self.seq_len - len(sequence))
        for i in range(self.batch_size):
            self.solution.append(copy.deepcopy(sequence))
        self.obj_value = self.calculate_obj_value(self.solution)

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

    def optimize_init(self):
        for i in range(self.batch_size):
            separate_tokens = self.separate_tokens[i]
            non_special_tokens = [token for token in separate_tokens if
                                  token not in self.tokenizer.all_special_ids]
            for j in range(1, self.longest_length):
                solutions = []
                objs = []
                for token in non_special_tokens:
                    solution = copy.deepcopy(self.solution)
                    sequence = solution[i]
                    sequence[j] = token
                    solution[i] = sequence
                    obj_value = self.calculate_obj_value(solution)
                    solutions.append(solution)
                    objs.append(obj_value)
                self.solution = solutions[objs.index(max(objs))]
                self.obj_value = max(objs)

    def optimize(self):
        for i in range(self.num_of_iter):
            for j in range(self.batch_size):
                separate_tokens = self.separate_tokens[j]
                non_cls_sep_tokens = [token for token in separate_tokens if
                                      token not in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]]
                for k in range(1, self.longest_length):
                    solutions = []
                    objs = []
                    for token in non_cls_sep_tokens:
                        solution = copy.deepcopy(self.solution)
                        sequence = solution[j]
                        sequence[k] = token
                        solution[j] = sequence
                        obj_value = self.calculate_obj_value(solution)
                        solutions.append(solution)
                        objs.append(obj_value)
                    self.solution = solutions[objs.index(max(objs))]
                    self.obj_value = max(objs)
            print("Iteration: ", i, " Obj value: ", self.obj_value, " Solution: ", self.solution)
        return self.solution, self.obj_value
