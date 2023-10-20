import torch
from torch.nn.functional import one_hot
import torch.nn.functional as F
import copy


class ContinuousOptimizer():
    # TODO: ATTENTION MASKS!!!
    def __init__(self, discrete_solution, client, server, tokenizer, device, client_grads, token_set,
                 token_embedding, labels, alpha=0.01, num_of_iterations=5000):
        self.discrete_solution = discrete_solution
        self.client = client
        self.server = server
        self.tokenizer = tokenizer
        self.device = device
        self.client_grads = client_grads
        # Remove token embedding layer gradients
        self.client_grads = self.client_grads[1:]
        self.num_of_iterations = num_of_iterations
        self.token_embedding = token_embedding
        self.dummy_embedding = None
        self.dummy_labels = None
        self.grads_diff = None
        self.continuous_solution = None
        self.token_set = token_set
        self.labels = labels
        self.server.eval()
        self.alpha = alpha

    # def calculate_continuous_solution_grads_diff(self, solution):
    #     dummy_x = torch.tensor(solution).to(self.device)
    #     dummy_y = torch.tensor(one_hot(dummy_x, num_classes=len(self.tokenizer)),
    #                            dtype=torch.float32).to(self.device)
    #     self.dummy_attention_mask = []
    #     for i in range(len(self.continuous_solution)):
    #         mask = []
    #         for j in range(len(self.continuous_solution[i])):
    #             if self.continuous_solution[i][j] == self.tokenizer.pad_token_id:
    #                 mask.append(0)
    #             else:
    #                 mask.append(1)
    #         self.dummy_attention_mask.append(mask)
    #
    #     dummy_outputs = self.server(inputs_embeds=self.dummy_embedding,
    #                                 attention_mask=torch.tensor(self.dummy_attention_mask).to(self.device))
    #     dummy_preds = dummy_outputs.logits
    #     loss_function = torch.nn.CrossEntropyLoss()
    #     shifted_preds = dummy_preds[..., :-1, :].contiguous()
    #     shifted_labels = dummy_y[..., 1:, :].contiguous()
    #     flatten_shifted_preds = shifted_preds.view(-1, shifted_preds.size(-1)).to(self.device)
    #     flatten_labels = shifted_labels.view(-1, shifted_labels.size(-1)).to(self.device)
    #     dummy_loss = loss_function(flatten_shifted_preds, flatten_labels)
    #     server_grads = torch.autograd.grad(dummy_loss, self.server.parameters(), create_graph=True)
    #     grads_diff = 0
    #     for gx, gy in zip(server_grads, self.client_grads):
    #         grads_diff += torch.norm(gx - gy, p=2)
    #     # beta = 1e-18
    #     # Calculate the perplexity of the solution as a second objective using the client model
    #     # self.client.eval()
    #     # with torch.no_grad():
    #     #     client_outputs = self.client(input_ids=dummy_x, labels=dummy_x.clone())
    #     #     client_loss = client_outputs.loss
    #     #     perplexity = torch.exp(client_loss)
    #     # Add the negative value such that bigger objective value means a better solution
    #     return grads_diff.item()

    def construct_input_embeddings(self):
        input_embeddings = []
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

    def set_up(self):
        self.construct_input_embeddings()
        self.dummy_attention_mask = []
        for i in range(len(self.discrete_solution)):
            mask = []
            for j in range(len(self.discrete_solution[i])):
                if self.discrete_solution[i][j] == self.tokenizer.pad_token_id:
                    mask.append(0)
                else:
                    mask.append(1)
            self.dummy_attention_mask.append(mask)
        self.token_set_embedding = []
        for token in self.token_set:
            self.token_set_embedding.append(self.token_embedding[token])
        self.token_set_embedding = torch.stack(self.token_set_embedding).to(self.device)

    def optimize(self):
        self.set_up()
        for i in range(self.num_of_iterations):
            if self.continuous_solution is not None:
                self.dummy_attention_mask = []
                for l in range(len(self.continuous_solution)):
                    mask = []
                    for k in range(len(self.continuous_solution[l])):
                        if self.continuous_solution[l][k] == self.tokenizer.pad_token_id:
                            mask.append(0)
                        else:
                            mask.append(1)
                    self.dummy_attention_mask.append(mask)

            dummy_outputs = self.server(inputs_embeds=self.dummy_embedding,
                                        attention_mask=torch.tensor(self.dummy_attention_mask).to(self.device))
            dummy_preds = dummy_outputs.logits
            loss_function = torch.nn.CrossEntropyLoss()
            dummy_loss = loss_function(dummy_preds, torch.LongTensor(self.labels).to(self.device))
            server_parameters = []
            for param in self.server.parameters():
                server_parameters.append(param)
            server_parameters = server_parameters[1:]
            server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
            optimizer = torch.optim.AdamW([self.dummy_embedding], lr=0.01)
            optimizer.zero_grad()
            grads_diff = 0
            for gx, gy in zip(server_grads, self.client_grads):
                grads_diff += torch.norm(gx - gy, p=2) + self.alpha * torch.norm(gx - gy, p=1)
                # grads_diff += torch.sum(gx * gy) / (torch.norm(gx, p=2) * torch.norm(gy, p=2))

            # # Add the embedding regularization term
            # embedding_regularization = (torch.norm(self.dummy_embedding, p=2) - torch.norm(self.token_set_embedding,
            #                                                                                p=2)) ** 2
            # grads_diff += alpha * embedding_regularization
            # grads_diff.backward(retain_graph=True)
            grads_diff.backward()
            optimizer.step()
            # Copy the dummy embeddings
            dummy_input_embedding_copy = self.dummy_embedding.detach().clone()
            # Calculate the similarity between each row of the dummy embeddings and the position tokens
            continuous_solution = []
            for sequence in dummy_input_embedding_copy:
                discrete_tokens = []
                for j in range(sequence.size()[0]):
                    token_embedding = sequence[j].unsqueeze(0)
                    similarity = torch.cosine_similarity(token_embedding, self.token_set_embedding)
                    # Get the index of the position token with the highest similarity
                    index = torch.argmax(similarity)
                    # Get the token id of the position token with the highest similarity
                    token_id = self.token_set[index]
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

        return -self.grads_diff, self.dummy_embedding, self.continuous_solution
