import torch
from torch.nn.functional import one_hot
import torch.nn.functional as F
import copy

from torch.optim import lr_scheduler


class ContinuousOptimizerWTE():
    def __init__(self, discrete_solution, client, server, tokenizer, device, client_grads,
                 token_embedding, labels, alpha, beta, longest_length, batch_size, seq_len, init_size, num_perm,
                 num_of_iterations=3000):
        self.discrete_solution = discrete_solution
        self.client = client
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.client_grads = client_grads
        # Remove token embedding layer gradients
        self.client_grads = self.client_grads
        self.num_of_iterations = num_of_iterations
        self.token_embedding = token_embedding
        self.dummy_embedding = None
        self.dummy_labels = None
        self.grads_diff = None
        self.continuous_solution = None
        self.labels = labels
        self.server.eval()
        self.alpha = alpha
        self.longest_length = longest_length
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.beta = beta
        self.init_size = init_size
        self.num_perm = num_perm
        self.avg_token_embedding_norm = self.token_embedding.norm(p=2, dim=1).mean()
        self.dummy_attention_mask = []
        for i in range(self.batch_size):
            mask = []
            for j in range(self.seq_len):
                if j < self.longest_length:
                    mask.append(1)
                else:
                    mask.append(0)
            self.dummy_attention_mask.append(mask)

    def construct_input_embeddings(self):
        input_embeddings = []
        if self.discrete_solution is not None:
            for i in range(len(self.discrete_solution)):
                input_embedding = []
                sequence = self.discrete_solution[i]
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
            self.dummy_embedding = torch.stack(input_embeddings).to(self.device).requires_grad_(True)
        else:
            embeddings = []
            objs = []
            for i in range(self.init_size):
                dummy_embedding = torch.randn(self.batch_size, self.longest_length - 2,
                                              self.token_embedding.shape[1]).to(self.device)
                cls_embedding = self.token_embedding[self.tokenizer.cls_token_id].repeat(self.batch_size, 1, 1)
                sep_embedding = self.token_embedding[self.tokenizer.sep_token_id].repeat(self.batch_size, 1, 1)
                pad_embedding = self.token_embedding[self.tokenizer.pad_token_id].repeat(self.batch_size,
                                                                                         self.seq_len - self.longest_length,
                                                                                         1)
                dummy_embedding = torch.cat((cls_embedding, dummy_embedding, sep_embedding, pad_embedding), dim=1)
                dummy_embedding /= torch.norm(dummy_embedding, dim=2, keepdim=True)
                dummy_embedding *= self.avg_token_embedding_norm
                dummy_embedding.to(self.device).requires_grad_(True)
                embeddings.append(dummy_embedding)

            for i in range(len(embeddings)):
                if i % 100 == 0:
                    print("Evaluating embedding: ", i)
                dummy_embedding = embeddings[i]
                dummy_outputs = self.server(inputs_embeds=dummy_embedding.to(self.device),
                                            attention_mask=torch.tensor(self.dummy_attention_mask).to(self.device))
                dummy_preds = dummy_outputs.logits
                loss_function = torch.nn.CrossEntropyLoss()
                dummy_loss = loss_function(dummy_preds, torch.LongTensor(self.labels).to(self.device))
                server_parameters = []
                for param in self.server.parameters():
                    if param.requires_grad:
                        server_parameters.append(param)

                server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
                grads_diff = 0
                for gx, gy in zip(server_grads, self.client_grads):
                    grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                    # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

                # Add the embedding regularization term
                embedding_regularization = (
                        dummy_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()
                grads_diff += self.beta * embedding_regularization
                objs.append(-grads_diff.item())

            best_embedding = embeddings[objs.index(max(objs))]
            best_embedding_real = best_embedding[:, 1:self.longest_length - 1, :]
            perms = []
            perm_objs = []
            for i in range(self.num_perm):
                idx = torch.randperm(self.longest_length - 2)
                perm = best_embedding_real[:, idx, :]
                cls_embedding = self.token_embedding[self.tokenizer.cls_token_id].repeat(self.batch_size, 1, 1)
                sep_embedding = self.token_embedding[self.tokenizer.sep_token_id].repeat(self.batch_size, 1, 1)
                pad_embedding = self.token_embedding[self.tokenizer.pad_token_id].repeat(self.batch_size,
                                                                                         self.seq_len - self.longest_length,
                                                                                         1)
                perm_embedding = torch.cat((cls_embedding, perm, sep_embedding, pad_embedding), dim=1)
                perm_embedding /= torch.norm(perm_embedding, dim=2, keepdim=True)
                perm_embedding *= self.avg_token_embedding_norm
                perm_embedding.to(self.device).requires_grad_(True)
                perms.append(perm_embedding)

            for i in range(len(perms)):
                if i % 100 == 0:
                    print("Evaluating perm: ", i)
                perm_embedding = perms[i]
                dummy_outputs = self.server(inputs_embeds=perm_embedding.to(self.device),
                                            attention_mask=torch.tensor(self.dummy_attention_mask).to(self.device))
                dummy_preds = dummy_outputs.logits
                loss_function = torch.nn.CrossEntropyLoss()
                dummy_loss = loss_function(dummy_preds, torch.LongTensor(self.labels).to(self.device))
                server_parameters = []
                for param in self.server.parameters():
                    if param.requires_grad:
                        server_parameters.append(param)

                server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
                grads_diff = 0
                for gx, gy in zip(server_grads, self.client_grads):
                    grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                    # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

                # Add the embedding regularization term
                embedding_regularization = (
                        perm_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()



                grads_diff += self.beta * embedding_regularization
                perm_objs.append(-grads_diff.item())

            self.dummy_embedding = perms[perm_objs.index(max(perm_objs))].clone().detach().to(self.device).requires_grad_(True)

            # self.dummy_embedding = torch.randn(self.batch_size, self.longest_length - 2,
            #                                    self.token_embedding.shape[1]).to(self.device)
            # cls_embedding = self.token_embedding[self.tokenizer.cls_token_id].repeat(self.batch_size, 1, 1)
            # sep_embedding = self.token_embedding[self.tokenizer.sep_token_id].repeat(self.batch_size, 1, 1)
            # pad_embedding = self.token_embedding[self.tokenizer.pad_token_id].repeat(self.batch_size,
            #                                                                          self.seq_len - self.longest_length,
            #                                                                          1)
            # self.dummy_embedding = torch.cat((cls_embedding, self.dummy_embedding, sep_embedding, pad_embedding),
            #                                  dim=1)
            # self.dummy_embedding /= torch.norm(self.dummy_embedding, dim=2, keepdim=True)
            # self.dummy_embedding *= self.avg_token_embedding_norm
            # self.dummy_embedding.to(self.device).requires_grad_(True)

    def set_up(self):
        self.construct_input_embeddings()

    def optimize(self):
        self.set_up()
        optimizer = torch.optim.AdamW([self.dummy_embedding], lr=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.89)
        for i in range(self.num_of_iterations):
            # if self.continuous_solution is not None:
            #     self.dummy_attention_mask = []
            #     for l in range(len(self.continuous_solution)):
            #         mask = []
            #         for k in range(len(self.continuous_solution[l])):
            #             if self.continuous_solution[l][k] == self.tokenizer.pad_token_id:
            #                 mask.append(0)
            #             else:
            #                 mask.append(1)
            #         self.dummy_attention_mask.append(mask)

            dummy_outputs = self.server(inputs_embeds=self.dummy_embedding,
                                        attention_mask=torch.tensor(self.dummy_attention_mask).to(self.device))
            dummy_preds = dummy_outputs.logits
            loss_function = torch.nn.CrossEntropyLoss()
            dummy_loss = loss_function(dummy_preds, torch.LongTensor(self.labels).to(self.device))
            server_parameters = []
            for param in self.server.parameters():
                if param.requires_grad:
                    server_parameters.append(param)

            server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
            # optimizer = torch.optim.AdamW([self.dummy_embedding], lr=0.01)
            optimizer.zero_grad()
            grads_diff = 0
            for gx, gy in zip(server_grads, self.client_grads):
                grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

            # Add the embedding regularization term
            embedding_regularization = (
                    self.dummy_embedding.norm(p=2, dim=2).mean() - self.avg_token_embedding_norm).square()
            grads_diff += self.beta * embedding_regularization

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
            grads_diff.backward()
            optimizer.param_groups[0]['params'][0] = self.dummy_embedding
            optimizer.step()
            scheduler.step()
            # Change back the CLS, SEP and PAD tokens to the original embeddings
            # Copy the dummy embeddings
            dummy_input_embedding_copy = self.dummy_embedding.detach().clone()
            for j in range(self.batch_size):
                dummy_input_embedding_copy[j, 0, :] = self.token_embedding[self.tokenizer.cls_token_id]
                dummy_input_embedding_copy[j, self.longest_length - 1, :] = self.token_embedding[
                    self.tokenizer.sep_token_id]
                for k in range(self.seq_len - self.longest_length):
                    dummy_input_embedding_copy[j, self.longest_length + k, :] = self.token_embedding[
                        self.tokenizer.pad_token_id]

            self.dummy_embedding = dummy_input_embedding_copy.detach().clone().to(self.device).requires_grad_(True)
            # Calculate the similarity between each row of the dummy embeddings and the position tokens
            continuous_solution = []
            for sequence in dummy_input_embedding_copy:
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
                print("Gradients difference: ", self.grads_diff)
                print("Current dummy input:")
                for j in range(len(self.continuous_solution)):
                    print(self.tokenizer.decode(self.continuous_solution[j]))

        return -self.grads_diff, self.dummy_embedding, self.continuous_solution,
