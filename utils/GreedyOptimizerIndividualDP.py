import torch
import copy
import numpy as np
import joblib


class GreedyOptimizerIndividualDP():
    def __init__(self, server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                 labels, start_tokens, longest_length, individual_lengths, separate_tokens, token_set, parallel, alpha=0.05,
                 num_of_solutions=1, num_of_iter=1, num_of_perms=2000, continuous_solution=None,
                 discrete_solution=None):
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.client_grads = client_grads
        self.num_of_solutions = num_of_solutions
        self.continuous_solution = continuous_solution
        self.discrete_solution = discrete_solution
        self.num_of_iter = num_of_iter
        self.result_file = result_file
        self.parallel = parallel
        self.solutions = None
        self.obj_values = None
        self.global_best_solution = None
        self.global_best_obj = None
        self.labels = labels
        self.start_tokens = start_tokens
        self.estimated_len = longest_length
        self.individual_lengths = individual_lengths
        self.non_special_token_set = [token for token in token_set if token not in self.tokenizer.all_special_ids]
        self.separate_tokens = separate_tokens
        self.alpha = alpha
        self.num_of_perms = num_of_perms

    def set_up(self):
        self.solutions = []
        perms = []
        objs = []
        perms.append(self.continuous_solution)
        objs.append(self.calculate_obj_value(self.continuous_solution))
        if self.discrete_solution is not None:
            perms.append(self.discrete_solution)
            objs.append(self.calculate_obj_value(self.discrete_solution))

        real_tokens = []
        for i in range(self.batch_size):
            real_tokens.append(torch.LongTensor(self.continuous_solution[i][1:self.individual_lengths[i] - 1]))
        for i in range(self.num_of_perms):
            if i % 100 == 0:
                print(f"Evaluating continuous perm {i}")
            perm = []
            for j in range(self.batch_size):
                idx = torch.randperm(self.individual_lengths[j] - 2)
                perm.append(real_tokens[j][idx].tolist())
            for j in range(self.batch_size):
                sequence = perm[j]
                sequence = [self.tokenizer.cls_token_id] + sequence + [self.tokenizer.sep_token_id]
                sequence = sequence + [self.tokenizer.pad_token_id] * (self.seq_len - len(sequence))
                perm[j] = sequence
            perms.append(perm)
            objs.append(self.calculate_obj_value(perm))

        # if self.discrete_solution is not None:
        #     real_tokens = torch.LongTensor(self.discrete_solution)[:, 1:self.estimated_len - 1]
        #     for i in range(self.num_of_perms):
        #         if i % 100 == 0:
        #             print(f"Evaluating discrete perm {i}")
        #         idx = torch.randperm(self.estimated_len - 2)
        #         perm = real_tokens[:, idx]
        #         perm = perm.tolist()
        #         for j in range(self.batch_size):
        #             sequence = perm[j]
        #             sequence = [self.tokenizer.cls_token_id] + sequence + [self.tokenizer.sep_token_id]
        #             sequence = sequence + [self.tokenizer.pad_token_id] * (self.seq_len - len(sequence))
        #             perm[j] = sequence
        #         perms.append(perm)
        #         objs.append(self.calculate_obj_value(perm))

        # Select the top num_of_solutions solutions from perms
        sorted_idx = np.argsort(objs)[::-1]
        for i in range(self.num_of_solutions):
            self.solutions.append(perms[sorted_idx[i]])

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
        # self.server.zero_grad()
        server_parameters = []
        for param in self.server.parameters():
            if param.requires_grad:
                server_parameters.append(param)
        server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
        server_grads = [grad / grad.norm() for grad in server_grads]
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
            separate_tokens = self.separate_tokens[j]
            length = self.individual_lengths[j]
            for k in range(1, length):
                for token in [token for token in separate_tokens if token != self.tokenizer.cls_token_id]:
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
            if not self.parallel:
                current_best_solutions = []
                for i in range(self.num_of_solutions):
                    current_best_solutions.append(self.optimize_one_solution(i))
            else:
                current_best_solutions = joblib.Parallel(n_jobs=2)(
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
            with open(self.result_file, "a", encoding="utf-8") as file:
                file.write(f"iter {l}:\n")
                for i in range(len(self.global_best_solution)):
                    file.write(f"{self.tokenizer.decode(self.global_best_solution[i], skip_special_tokens=True)}\n")
                file.write(f"{self.global_best_obj}\n")

            for i in range(len(self.global_best_solution)):
                print(self.tokenizer.decode(self.global_best_solution[i]))
        return self.global_best_solution, self.global_best_obj
