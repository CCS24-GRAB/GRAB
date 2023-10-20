import torch
import joblib
import copy
import random
from torch.nn.functional import one_hot
import numpy as np


class GreedyOptimizer():
    def __init__(self, client, server, tokenizer, device, seq_len, batch_size, client_grads, token_set, result_file,
                 labels, start_tokens, longest_length, alpha=0.01,
                 num_of_solutions=1, num_of_iter=1, continuous_solution=None, discrete_solution=None):
        self.client = client
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.client_grads = client_grads
        self.token_set = token_set
        self.num_of_solutions = num_of_solutions
        self.continuous_solution = continuous_solution
        self.discrete_solution = discrete_solution
        self.num_of_iter = num_of_iter
        self.result_file = result_file
        self.solutions = None
        self.obj_values = None
        self.global_best_solution = None
        self.global_best_obj = None
        self.labels = labels
        self.start_tokens = start_tokens
        self.estimated_len = longest_length
        self.non_special_token_set = [token for token in token_set if token not in self.tokenizer.all_special_ids]
        self.alpha = alpha

    def set_up(self):
        self.solutions = []
        for i in range(self.num_of_solutions):
            solution = []
            for j in range(self.batch_size):
                sequence = [self.tokenizer.cls_token_id, random.choice(self.start_tokens)]
                for k in range(self.estimated_len - 3):
                    token = random.choice(self.non_special_token_set)
                    sequence.append(token)
                sequence = sequence + [self.tokenizer.sep_token_id]
                # Pad the sequence to seq_len
                sequence = sequence + [self.tokenizer.pad_token_id] * (self.seq_len - len(sequence))
                solution.append(sequence)
            # solution = self.move_end_and_pad_tokens(solution)
            self.solutions.append(solution)

        if self.continuous_solution is not None:
            self.solutions[0] = copy.deepcopy(self.continuous_solution)
        if self.discrete_solution is not None:
            self.solutions[-1] = copy.deepcopy(self.discrete_solution)

        self.calculate_population_obj_values()
        self.global_best_solution = self.solutions[np.argmax(self.obj_values)]
        self.global_best_obj = np.max(self.obj_values)

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

    def calculate_population_obj_values(self):
        self.obj_values = []
        for solution in self.solutions:
            self.obj_values.append(self.calculate_obj_value(solution))
        return self.obj_values

    def optimize_one_solution(self, index):
        print(f"optimizing solution {index}")
        current_best_obj = self.obj_values[index]
        current_best_solution = copy.deepcopy(self.solutions[index])
        # if self.start_tokens == self.non_special_token_set:
        #     start = 1
        # else:
        #     start = 2
        for j in range(self.batch_size):
            for k in range(1, self.estimated_len):
                for token in [token for token in self.token_set if token != self.tokenizer.cls_token_id]:
                    sequence = copy.deepcopy(current_best_solution[j])
                    sequence[k] = token
                    temp_solution = copy.deepcopy(current_best_solution)
                    temp_solution[j] = sequence
                    temp_obj = self.calculate_obj_value(temp_solution)
                    if temp_obj > current_best_obj:
                        current_best_obj = temp_obj
                        current_best_solution = copy.deepcopy(temp_solution)
            return current_best_solution

    # def move_end_and_pad_tokens(self, solution):
    #     for i in range(len(solution)):
    #         sequence = solution[i]
    #         if self.tokenizer.sep_token_id not in sequence:
    #             sequence[-1] = self.tokenizer.sep_token_id
    #         # Move the end token to the end of the sequence
    #         for k in range(len(sequence)):
    #             if sequence[k] == self.tokenizer.sep_token_id:
    #                 sequence.append(sequence.pop(k))
    #         # Move the pad token to the end of the sequence
    #         for k in range(len(sequence)):
    #             if sequence[k] == self.tokenizer.pad_token_id:
    #                 sequence.append(sequence.pop(k))
    #         solution[i] = sequence
    #     return solution

    def optimize(self):
        print("Begin greedy optimisation")
        self.set_up()
        self.calculate_population_obj_values()
        for l in range(self.num_of_iter):
            # for i in range(len(self.solutions)):
            #     current_best_solution = self.optimize_one_solution(i)
            #     self.solutions[i] = copy.deepcopy(current_best_solution)
            current_best_solutions = joblib.Parallel(n_jobs=8)(
                joblib.delayed(self.optimize_one_solution)(i) for i in range(self.num_of_solutions))
            self.solutions = copy.deepcopy(current_best_solutions)
            # for i in range(len(self.solutions)):
            #     self.solutions[i] = self.move_end_and_pad_tokens(self.solutions[i])
            self.calculate_population_obj_values()
            local_best_solution = self.solutions[np.argmax(self.obj_values)]
            local_best_solution_obj = np.max(self.obj_values)
            if self.global_best_solution is None:
                self.global_best_solution = copy.deepcopy(local_best_solution)
                self.global_best_obj = local_best_solution_obj
            elif local_best_solution_obj > self.global_best_obj:
                self.global_best_solution = copy.deepcopy(local_best_solution)
                self.global_best_obj = local_best_solution_obj
            print(f"iter {l}, global best obj {self.global_best_obj}")
            print(f"global best solution:")
            with open(self.result_file, "a") as file:
                file.write(f"iter {l}:\n")
                for i in range(len(self.global_best_solution)):
                    file.write(f"{self.tokenizer.decode(self.global_best_solution[i], skip_special_tokens=True)}\n")
                file.write(f"{self.global_best_obj}\n")

            for i in range(len(self.global_best_solution)):
                print(self.tokenizer.decode(self.global_best_solution[i]))
        return self.global_best_solution, self.global_best_obj
