import torch
from torch.nn.functional import one_hot
import torch.nn.functional as F
import copy
from utils import *
from torch.optim import lr_scheduler
import gc


class ContinuousOptimizerIndividualDP:
    def __init__(self, discrete_solution, server, tokenizer, device, client_grads,
                 token_embedding, labels, alpha, beta, individual_lengths, batch_size, seq_len, init_size, num_perm, lr,
                 lr_decay_type, simreg=100,
                 optimize_dropout_mask=True,
                 num_of_iterations=3000, observe_embedding=False):
        self.discrete_solution = discrete_solution
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.client_grads = copy.deepcopy(client_grads)
        self.num_of_iterations = num_of_iterations
        self.token_embedding = token_embedding
        self.dummy_embedding = None
        self.dummy_labels = None
        self.grads_diff = None
        self.continuous_solution = None
        self.labels = labels
        self.labels = torch.LongTensor(self.labels).to(self.device)
        self.alpha = alpha
        self.individual_lengths = individual_lengths
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.beta = beta
        self.init_size = init_size
        self.num_perm = num_perm
        self.lr = lr
        self.lr_decay_type = lr_decay_type
        self.simreg = simreg
        self.avg_token_embedding_norm = self.token_embedding.norm(p=2, dim=1).mean()
        self.dummy_attention_mask = []
        for i in range(self.batch_size):
            mask = []
            length = self.individual_lengths[i]
            for j in range(self.seq_len):
                if j < length:
                    mask.append(1)
                else:
                    mask.append(0)
            self.dummy_attention_mask.append(mask)
        self.dummy_attention_mask = torch.tensor(self.dummy_attention_mask).to(self.device)
        self.optimize_dropout_mask = optimize_dropout_mask

        self.dropout_masks = []
        for name, module in self.server.named_modules():
            if isinstance(module, CustomDropoutLayer):
                self.dropout_masks.append(module.mask)

        self.observe_embedding = observe_embedding
        if self.observe_embedding:
            self.client_grads = self.client_grads[1:]

    def construct_input_embeddings(self):
        if self.discrete_solution is not None:
            real_tokens = []
            for i in range(self.batch_size):
                real_tokens.append(torch.LongTensor(self.discrete_solution[i][1:self.individual_lengths[i] - 1]))
            perms = []
            objs = []
            perms.append(self.discrete_solution)
            for i in range(self.num_perm):
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

            perms_embeddings = []
            for i in range(len(perms)):
                perm = perms[i]
                input_embeddings = []
                for j in range(len(perm)):
                    input_embedding = []
                    sequence = perm[j]
                    for token in sequence:
                        if input_embedding == []:
                            input_embedding = torch.reshape(self.token_embedding[token].clone().detach(),
                                                            (1, self.token_embedding[token].shape[0]))
                        else:
                            input_embedding = torch.cat(
                                (input_embedding,
                                 torch.reshape(self.token_embedding[token].clone().detach(),
                                               (1, self.token_embedding[token].shape[0]))),
                                dim=0)
                    input_embeddings.append(input_embedding)
                input_embeddings = torch.stack(input_embeddings)
                perms_embeddings.append(input_embeddings)

            for i in range(len(perms_embeddings)):
                if i % 100 == 0:
                    print(f"Evaluating discrete perm {i}")
                perm_embedding = perms_embeddings[i].to(self.device).requires_grad_(True)
                perm_outputs = self.server(inputs_embeds=perm_embedding,
                                           attention_mask=self.dummy_attention_mask)
                perm_preds = perm_outputs.logits
                loss_function = torch.nn.CrossEntropyLoss()
                perm_loss = loss_function(perm_preds, self.labels)
                # self.server.zero_grad()
                server_parameters = []
                for param in self.server.parameters():
                    if param.requires_grad:
                        server_parameters.append(param)

                if self.observe_embedding:
                    server_parameters = server_parameters[1:]

                server_grads = torch.autograd.grad(perm_loss, server_parameters, create_graph=True)
                # Normalize the gradients
                server_grads = [grad / grad.norm() for grad in server_grads]

                grads_diff = 0
                for gx, gy in zip(server_grads, self.client_grads):
                    grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                    # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

                # Add the embedding regularization term
                embedding_regularization = (
                        perm_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()
                # Add the similarity regularization term
                valid_embedding = perm_embedding[:, 1:max(self.individual_lengths) - 1, :].clone().detach()
                valid_embedding_mean = valid_embedding.mean(dim=1)
                # Calculate pairwise cosine similarity between rows
                similarity_matrix = F.cosine_similarity(valid_embedding_mean.unsqueeze(1),
                                                        valid_embedding_mean.unsqueeze(0), dim=-1).to(self.device)
                # Exclude the similarity of rows with themselves
                similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(self.device)
                non_zero_similarity = similarity_matrix[similarity_matrix > 0]
                overall_similarity = non_zero_similarity.mean()
                if self.batch_size == 1:
                    similarity_regularization = 0
                else:
                    similarity_regularization = overall_similarity.item()
                grads_diff += self.beta * embedding_regularization
                grads_diff += self.simreg * similarity_regularization
                objs.append(-grads_diff.item())

            self.dummy_embedding = perms_embeddings[objs.index(max(objs))].clone().detach().to(
                self.device).requires_grad_(True)
            del perms_embeddings
            gc.collect()
            torch.cuda.empty_cache()

        else:
            embeddings = []
            objs = []
            for i in range(self.init_size):
                dummy_embedding = torch.randn(self.batch_size, max(self.individual_lengths) - 2,
                                              self.token_embedding.shape[1]).clone().detach().cpu()
                cls_embedding = self.token_embedding[self.tokenizer.cls_token_id].repeat(self.batch_size, 1,
                                                                                         1).clone().detach().cpu()
                sep_embedding = self.token_embedding[self.tokenizer.sep_token_id].repeat(self.batch_size, 1,
                                                                                         1).clone().detach().cpu()
                pad_embedding = self.token_embedding[self.tokenizer.pad_token_id].repeat(self.batch_size,
                                                                                         self.seq_len - max(
                                                                                             self.individual_lengths),
                                                                                         1).clone().detach().cpu()
                dummy_embedding = torch.cat((cls_embedding, dummy_embedding, sep_embedding, pad_embedding), dim=1)
                dummy_embedding /= torch.norm(dummy_embedding, dim=2, keepdim=True)
                dummy_embedding *= self.avg_token_embedding_norm.clone().detach().cpu()
                embeddings.append(dummy_embedding)

            for i in range(len(embeddings)):
                if i % 100 == 0:
                    print("Evaluating init embedding: ", i)
                dummy_embedding = embeddings[i].to(self.device).requires_grad_(True)
                dummy_outputs = self.server(inputs_embeds=dummy_embedding,
                                            attention_mask=self.dummy_attention_mask)
                dummy_preds = dummy_outputs.logits
                loss_function = torch.nn.CrossEntropyLoss()
                dummy_loss = loss_function(dummy_preds, self.labels)
                # self.server.zero_grad()
                server_parameters = []
                for param in self.server.parameters():
                    if param.requires_grad:
                        server_parameters.append(param)
                if self.observe_embedding:
                    server_parameters = server_parameters[1:]
                server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
                server_grads = [grad / grad.norm() for grad in server_grads]
                grads_diff = 0
                for gx, gy in zip(server_grads, self.client_grads):
                    grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                    # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

                # Add the embedding regularization term
                embedding_regularization = (
                        dummy_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()
                # Add the similarity regularization term
                valid_embedding = dummy_embedding[:, 1:max(self.individual_lengths) - 1, :].clone().detach()
                valid_embedding_mean = valid_embedding.mean(dim=1)
                # Calculate pairwise cosine similarity between rows
                similarity_matrix = F.cosine_similarity(valid_embedding_mean.unsqueeze(1),
                                                        valid_embedding_mean.unsqueeze(0), dim=-1).to(self.device)
                # Exclude the similarity of rows with themselves
                similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(self.device)
                non_zero_similarity = similarity_matrix[similarity_matrix > 0]
                overall_similarity = non_zero_similarity.mean()
                if self.batch_size == 1:
                    similarity_regularization = 0
                else:
                    similarity_regularization = overall_similarity.item()
                grads_diff += self.beta * embedding_regularization
                grads_diff += self.simreg * similarity_regularization
                objs.append(-grads_diff.item())

            best_embedding = embeddings[objs.index(max(objs))]
            best_embedding_real = best_embedding[:, 1:max(self.individual_lengths) - 1, :]
            perms = []
            perm_objs = []
            for i in range(self.num_perm):
                idx = torch.randperm(max(self.individual_lengths) - 2)
                perm = best_embedding_real[:, idx, :]
                cls_embedding = self.token_embedding[self.tokenizer.cls_token_id].repeat(self.batch_size, 1,
                                                                                         1).clone().detach().cpu()
                sep_embedding = self.token_embedding[self.tokenizer.sep_token_id].repeat(self.batch_size, 1,
                                                                                         1).clone().detach().cpu()
                pad_embedding = self.token_embedding[self.tokenizer.pad_token_id].repeat(self.batch_size,
                                                                                         self.seq_len - max(
                                                                                             self.individual_lengths),
                                                                                         1).clone().detach().cpu()
                perm_embedding = torch.cat((cls_embedding, perm, sep_embedding, pad_embedding), dim=1)
                perm_embedding /= torch.norm(perm_embedding, dim=2, keepdim=True)
                perm_embedding *= self.avg_token_embedding_norm.clone().detach().cpu()
                perms.append(perm_embedding)

            for i in range(len(perms)):
                if i % 100 == 0:
                    print("Evaluating perm: ", i)
                perm_embedding = perms[i].to(self.device).requires_grad_(True)
                dummy_outputs = self.server(inputs_embeds=perm_embedding,
                                            attention_mask=self.dummy_attention_mask)
                dummy_preds = dummy_outputs.logits
                loss_function = torch.nn.CrossEntropyLoss()
                dummy_loss = loss_function(dummy_preds, self.labels)
                # self.server.zero_grad()
                server_parameters = []
                for param in self.server.parameters():
                    if param.requires_grad:
                        server_parameters.append(param)
                if self.observe_embedding:
                    server_parameters = server_parameters[1:]
                server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
                server_grads = [grad / grad.norm() for grad in server_grads]
                grads_diff = 0
                for gx, gy in zip(server_grads, self.client_grads):
                    grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                    # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

                # Add the embedding regularization term
                embedding_regularization = (
                        perm_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()

                # Add the similarity regularization term
                valid_embedding = perm_embedding[:, 1:max(self.individual_lengths) - 1, :].clone().detach()
                valid_embedding_mean = valid_embedding.mean(dim=1)
                # Calculate pairwise cosine similarity between rows
                similarity_matrix = F.cosine_similarity(valid_embedding_mean.unsqueeze(1),
                                                        valid_embedding_mean.unsqueeze(0), dim=-1).to(self.device)
                # Exclude the similarity of rows with themselves
                similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(self.device)
                non_zero_similarity = similarity_matrix[similarity_matrix > 0]
                overall_similarity = non_zero_similarity.mean()
                if self.batch_size == 1:
                    similarity_regularization = 0
                else:
                    similarity_regularization = overall_similarity.item()
                grads_diff += self.beta * embedding_regularization
                grads_diff += self.simreg * similarity_regularization
                perm_objs.append(-grads_diff.item())

            self.dummy_embedding = perms[perm_objs.index(max(perm_objs))].clone().detach().to(
                self.device).requires_grad_(True)
            del perms
            gc.collect()
            torch.cuda.empty_cache()

    def set_up(self):
        self.construct_input_embeddings()

    def optimize(self):
        self.set_up()
        optimizing_tensors = []
        optimizing_tensors.append(self.dummy_embedding)
        if self.optimize_dropout_mask:
            for mask in self.dropout_masks:
                optimizing_tensors.append(mask)

        optimizer = torch.optim.AdamW(optimizing_tensors, lr=self.lr)

        if self.lr_decay_type == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.89)
        elif self.lr_decay_type == "LambdaLR":
            def lr_lambda(current_step: int):
                return max(0.0, float(self.num_of_iterations - current_step) / float(max(1, self.num_of_iterations)))
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for i in range(self.num_of_iterations):

            dummy_outputs = self.server(inputs_embeds=self.dummy_embedding,
                                        attention_mask=self.dummy_attention_mask)
            dummy_preds = dummy_outputs.logits
            loss_function = torch.nn.CrossEntropyLoss()
            dummy_loss = loss_function(dummy_preds, self.labels)
            # self.server.zero_grad()
            server_parameters = []
            for param in self.server.parameters():
                if param.requires_grad:
                    server_parameters.append(param)
            if self.observe_embedding:
                server_parameters = server_parameters[1:]
            server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
            # print("Before adding noise: ", server_grads[0])
            server_grads = [grad / grad.norm() for grad in server_grads]
            # print("After adding noise: ", server_grads[0])
            # optimizer = torch.optim.AdamW([self.dummy_embedding], lr=0.01)
            optimizer.zero_grad()
            grads_diff = 0
            for gx, gy in zip(server_grads, self.client_grads):
                grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)

            # Add the embedding regularization term
            embedding_regularization = (
                    self.dummy_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()
            # Add the similarity regularization term
            valid_embedding = self.dummy_embedding[:, 1:max(self.individual_lengths) - 1, :].clone().detach()
            valid_embedding_mean = valid_embedding.mean(dim=1)
            # Calculate pairwise cosine similarity between rows
            similarity_matrix = F.cosine_similarity(valid_embedding_mean.unsqueeze(1),
                                                    valid_embedding_mean.unsqueeze(0), dim=-1).to(self.device)
            # Exclude the similarity of rows with themselves
            similarity_matrix = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(self.device)
            non_zero_similarity = similarity_matrix[similarity_matrix > 0]
            overall_similarity = non_zero_similarity.mean()
            if self.batch_size == 1:
                similarity_regularization = 0
            else:
                similarity_regularization = overall_similarity.item()
            grads_diff += self.beta * embedding_regularization
            grads_diff += self.simreg * similarity_regularization

            # # Add the embedding regularization term
            # embedding_regularization = (torch.norm(self.dummy_embedding, p=2) - torch.norm(self.token_set_embedding,
            #                                                                                p=2)) ** 2
            # grads_diff += alpha * embedding_regularization
            # grads_diff.backward(retain_graph=True)
            # Add the embedding regularization term
            # dummy_embedding_norm = 0
            # for j in range(self.dummy_embedding.size()[0]):
            #     for k in range(self.dummy_embedding.size()[1]):
            #         dummy_embedding_norm += torch.norm(self.dummy_embedding[j][k], p=2)
            # dummy_embedding_norm = dummy_embedding_norm / (self.dummy_embedding.size()[0] * self.dummy_embedding.size()[1])
            # token_embedding_norm = 0
            # for j in range(self.token_embedding.size()[0]):
            #     token_embedding_norm += torch.norm(self.token_embedding[j], p=2)
            # token_embedding_norm = token_embedding_norm / self.token_embedding.size()[0]
            # embedding_regularization = beta * (dummy_embedding_norm - token_embedding_norm) ** 2
            # grads_diff += embedding_regularization
            grads_diff.backward(retain_graph=True)
            optimizer.param_groups[0]['params'][0] = self.dummy_embedding
            optimizer.step()
            scheduler.step()
            # Clip the dropout masks to range [0, 1]
            for mask in self.dropout_masks:
                mask.data.clamp_(0, 1)
            # Change back the CLS, SEP and PAD tokens to the original embeddings
            with torch.no_grad():
                for j in range(self.batch_size):
                    length = self.individual_lengths[j]
                    self.dummy_embedding[j, 0, :] = self.token_embedding[self.tokenizer.cls_token_id]
                    self.dummy_embedding[j, length - 1, :] = self.token_embedding[
                        self.tokenizer.sep_token_id]
                    for k in range(self.seq_len - length):
                        self.dummy_embedding[j, length + k, :] = self.token_embedding[
                            self.tokenizer.pad_token_id]

            # Calculate the similarity between each row of the dummy embeddings and the position tokens
            continuous_solution = []
            for sequence in self.dummy_embedding:
                discrete_tokens = []
                for j in range(sequence.size()[0]):
                    token_embedding = sequence[j].unsqueeze(0)
                    similarity = torch.cosine_similarity(token_embedding, self.token_embedding)
                    # Get the index of the position token with the highest similarity
                    index = torch.argmax(similarity)
                    # Get the token id of the position token with the highest similarity
                    token_id = index.item()
                    discrete_tokens.append(token_id)
                continuous_solution.append(discrete_tokens)
            self.grads_diff = grads_diff.item()
            self.continuous_solution = copy.deepcopy(continuous_solution)
            if i % 100 == 0:
                print("Iteration: ", i)
                print("Learning rate: ", optimizer.param_groups[0]['lr'])
                print("Gradients difference: ", self.grads_diff)
                print("Current dummy input:")
                for j in range(len(self.continuous_solution)):
                    print(self.tokenizer.decode(self.continuous_solution[j]))

        # return optimizer.param_groups[0][
        #     'lr'], self.server, -self.grads_diff, self.dummy_embedding, self.continuous_solution
        # for param in self.server.parameters():
        #     param.detach_()
        self.server.zero_grad()
        return copy.deepcopy(self.server), -self.grads_diff, self.continuous_solution
        # with torch.no_grad():
        #     return self.server, -self.grads_diff, self.continuous_solution
