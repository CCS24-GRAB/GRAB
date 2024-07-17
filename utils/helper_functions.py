from collections import OrderedDict
from tqdm import tqdm
import random
import torch
import numpy as np


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_epoch(model, train_loader, device):
    """
    Train a client model on local data for a complete epoch.
    Args:
        model: The model to be trained
        train_loader: The training data loader
        device: The device to run the model on

    Returns:
        The average training loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # TODO: This is to disable dropout and batch normalization.
    model.eval()
    train_loop = tqdm(train_loader, leave=False)
    epoch_train_loss = 0
    for batch in train_loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        train_loss = train_outputs.loss
        train_loss.backward()
        optimizer.step()
        epoch_train_loss += train_loss.item()
        train_loop.set_description(f"Training loss: {train_loss.item()}")
    average_epoch_loss = epoch_train_loss / len(train_loop)
    print(f"Epoch average training loss: {average_epoch_loss}")
    return average_epoch_loss


def train_batch(model, batch, device):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss and the attention scores
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def train_batch_with_noise(model, batch, device, noise_level):
    parameters = []
    for param in model.parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    for param in parameters:
        grad_norm = param.grad.norm()
        param.grad = (param.grad / grad_norm) + torch.randn(param.grad.shape).to(
            device) * noise_level
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def train_batch_with_prune(model, batch, device, prune):
    parameters = []
    for param in model.parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    for param in parameters:
        param.grad = param.grad * (torch.rand(param.grad.shape).to(device) > prune).float()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


# def share_gradient(self, noise_scale=NOISE_SCALE, agr=False):
#     """
#     Participants share gradient to the aggregator
#     :return: None
#     """
#     gradient = self.get_epoch_gradient()
#     gradient, indices = select_by_threshold(gradient, GRADIENT_EXCHANGE_RATE, GRADIENT_SAMPLE_THRESHOLD)
#     noise = torch.randn(gradient.size()).to(DEVICE)
#     noise = (noise / noise.norm()) * noise_scale * gradient.norm()
#     # print("gradient norm before add noise {}".format(gradient.norm()), end = "")
#     gradient += noise
#     # print("gradient norm after add noise {}".format(gradient.norm()))
#     if agr:
#         self.aggregator.agr_loss_gradient_collect(gradient, indices)
#     else:
#         self.aggregator.collect(gradient, indices=indices, source=self.participant_index)
#     return gradient
def test_batch(model, batch, device):
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    test_loss = test_outputs.loss
    return test_loss.item(), test_outputs.logits


def train_batch_with_dropout(model, batch, device):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss and the attention scores
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def train_batch_with_frozen_weights_and_noise(model, batch, device, noise_level):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss and the attention scores
    """
    parameters = []
    for param in model.parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    for param in parameters:
        grad_norm = param.grad.norm()
        param.grad = (param.grad / grad_norm) + torch.randn(param.grad.shape).to(
            device) * noise_level
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def train_batch_with_frozen_weights_and_prune(model, batch, device, prune):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss and the attention scores
    """
    parameters = []
    for param in model.parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    for param in parameters:
        param.grad = param.grad * (torch.rand(param.grad.shape).to(device) > prune).float()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def aggregate_gradients(server, client_gradients):
    aggregated_gradients = {}
    for name, param in server.named_parameters():
        aggregated_gradients[name] = torch.zeros(param.shape)

    for j in range(len(client_gradients)):
        single_client_gradient = client_gradients[j]
        for name, gradient in single_client_gradient.items():
            aggregated_gradients[name] += gradient.clone().detach().cpu()

    for key in aggregated_gradients.keys():
        aggregated_gradients[key] /= len(client_gradients)
    return aggregated_gradients


def update_server(server, aggregated_gradients, learning_rate):
    # Update the server model with the aggregated gradients and learning rate
    with torch.no_grad():
        for name, param in server.named_parameters():
            param.data = param.data.clone().detach().cpu() - learning_rate * aggregated_gradients[name]

    # for name, param in server.named_parameters():
    #     assert torch.all(torch.eq(param.data, prev_server_params[name].data - learning_rate * aggregated_gradients[name]))
    #
    # for name, param in server.named_parameters():
    #     assert torch.all(torch.eq(aggregated_gradients[name], (param.data - prev_server_params[name].data) / (-learning_rate)))
    return server


def calculate_aggregated_gradients_from_servers(server, prev_server_params, learning_rate):
    aggregated_gradients = {}
    for name, param in server.named_parameters():
        aggregated_gradients[name] = (param.data - prev_server_params[name].data) / (-learning_rate)
    return aggregated_gradients


def test(model, test_loader, device):
    """
    Test the server model after aggregation.
    Args:
        model: The server model to be tested
        test_loader: The testing data loader

    Returns:
        The average testing loss
    """
    model.eval()
    test_loop = tqdm(test_loader, leave=False)
    epoch_test_loss = 0
    with torch.no_grad():
        for batch in test_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss = test_outputs.loss
            epoch_test_loss += test_loss.item()
            test_loop.set_description(f"Test loss: {test_loss.item()}")
        average_epoch_loss = epoch_test_loss / len(test_loop)
    return average_epoch_loss


def train_batch_with_frozen_weights(model, batch, device):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss and the attention scores
    """
    parameters = []
    for param in model.parameters():
        if param.requires_grad:
            parameters.append(param)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


def reorganise_solution(solution, longest_length, sequence_length, tokenizer):
    for i in range(len(solution)):
        sequence = solution[i]
        non_special_ids = [id for id in sequence if id not in tokenizer.all_special_ids]
        non_special_ids = [tokenizer.cls_token_id] + non_special_ids
        if len(non_special_ids) < longest_length:
            non_special_ids += [tokenizer.sep_token_id]
            non_special_ids += [tokenizer.pad_token_id] * (sequence_length - len(non_special_ids))
        else:
            non_special_ids = non_special_ids[:longest_length]
            non_special_ids += [tokenizer.sep_token_id]
            non_special_ids += [tokenizer.pad_token_id] * (sequence_length - len(non_special_ids))
        solution[i] = non_special_ids
    return solution


def set_parameters(model, parameters):
    """
    Set the parameters of a model.
    Args:
        model: A neural network models with parameters.
        parameters: A list of parameters for the model.

    Returns:
        The model with the new parameters.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def get_parameters(model):
    """
    Get the parameters of a model.
    Args:
        model: A neural network models with parameters.

    Returns:
        The parameters of the model.
    """
    params_dict = model.state_dict()
    params = []
    for key in params_dict.keys():
        params.append(params_dict[key].cpu().numpy())
    return params


def calculate_discrete_obj(solution, labels, server, client_grads, device, tokenizer, alpha):
    dummy_x = torch.tensor(solution).to(device)
    dummy_y = torch.LongTensor(labels).to(device)
    dummy_attention_mask = []
    for i in range(len(solution)):
        mask = []
        for j in range(len(solution[i])):
            if solution[i][j] == tokenizer.pad_token_id:
                mask.append(0)
            else:
                mask.append(1)
        dummy_attention_mask.append(mask)
    dummy_attention_mask = torch.tensor(dummy_attention_mask).to(device)
    dummy_outputs = server(input_ids=dummy_x,
                           attention_mask=dummy_attention_mask)
    dummy_preds = dummy_outputs.logits
    loss_function = torch.nn.CrossEntropyLoss()
    dummy_loss = loss_function(dummy_preds, dummy_y)
    server_parameters = []
    for param in server.parameters():
        if param.requires_grad:
            server_parameters.append(param)
    server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
    grads_diff = 0
    for gx, gy in zip(server_grads, client_grads):
        grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
    return -grads_diff.item()

def calculate_discrete_obj_label(solution, labels, server, client_grads, device, tokenizer, alpha):
    dummy_x = torch.tensor(solution).to(device)
    dummy_attention_mask = []
    for i in range(len(solution)):
        mask = []
        for j in range(len(solution[i])):
            if solution[i][j] == tokenizer.pad_token_id:
                mask.append(0)
            else:
                mask.append(1)
        dummy_attention_mask.append(mask)
    dummy_attention_mask = torch.tensor(dummy_attention_mask).to(device)
    dummy_outputs = server(input_ids=dummy_x,
                           attention_mask=dummy_attention_mask)
    dummy_preds = dummy_outputs.logits
    loss_function = torch.nn.CrossEntropyLoss()
    dummy_loss = loss_function(dummy_preds, labels)
    server_parameters = []
    for param in server.parameters():
        if param.requires_grad:
            server_parameters.append(param)
    server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
    grads_diff = 0
    for gx, gy in zip(server_grads, client_grads):
        grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
    return -grads_diff.item()

def calculate_discrete_obj_prune(solution, labels, server, client_grads, device, tokenizer, alpha, prune_mask):
    dummy_x = torch.tensor(solution).to(device)
    dummy_y = torch.LongTensor(labels).to(device)
    dummy_attention_mask = []
    for i in range(len(solution)):
        mask = []
        for j in range(len(solution[i])):
            if solution[i][j] == tokenizer.pad_token_id:
                mask.append(0)
            else:
                mask.append(1)
        dummy_attention_mask.append(mask)
    dummy_attention_mask = torch.tensor(dummy_attention_mask).to(device)
    dummy_outputs = server(input_ids=dummy_x,
                           attention_mask=dummy_attention_mask)
    dummy_preds = dummy_outputs.logits
    loss_function = torch.nn.CrossEntropyLoss()
    dummy_loss = loss_function(dummy_preds, dummy_y)
    server_parameters = []
    for param in server.parameters():
        if param.requires_grad:
            server_parameters.append(param)

    server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
    server_grads = list(server_grads)
    for i in range(len(server_grads)):
        server_grads[i] = server_grads[i] * prune_mask[i]
    grads_diff = 0
    for gx, gy in zip(server_grads, client_grads):
        grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
    return -grads_diff.item()


def calculate_discrete_obj_DP(solution, labels, server, client_grads, device, tokenizer, alpha):
    dummy_x = torch.tensor(solution).to(device)
    dummy_y = torch.LongTensor(labels).to(device)
    dummy_attention_mask = []
    for i in range(len(solution)):
        mask = []
        for j in range(len(solution[i])):
            if solution[i][j] == tokenizer.pad_token_id:
                mask.append(0)
            else:
                mask.append(1)
        dummy_attention_mask.append(mask)
    dummy_attention_mask = torch.tensor(dummy_attention_mask).to(device)
    dummy_outputs = server(input_ids=dummy_x,
                           attention_mask=dummy_attention_mask)
    dummy_preds = dummy_outputs.logits
    loss_function = torch.nn.CrossEntropyLoss()
    dummy_loss = loss_function(dummy_preds, dummy_y)
    server_parameters = []
    for param in server.parameters():
        if param.requires_grad:
            server_parameters.append(param)
    server_grads = torch.autograd.grad(dummy_loss, server_parameters, create_graph=True)
    server_grads = [grad / grad.norm() for grad in server_grads]
    grads_diff = 0
    for gx, gy in zip(server_grads, client_grads):
        grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
    return -grads_diff.item()


class CustomDropoutLayer(torch.nn.Module):
    def __init__(self, shape, device, p=0.1):
        super(CustomDropoutLayer, self).__init__()
        self.mask = None
        self.p = p
        self.shape = shape
        self.device = device

    def init_mask(self):
        self.mask = torch.bernoulli(torch.ones(self.shape) * (1 - self.p)).detach().to(self.device).requires_grad_(True)

    def forward(self, x):
        if self.training:
            x = x * self.mask / (1 - self.p)
        return x


class CustomDropoutLayerGetShape(torch.nn.Module):
    def __init__(self, p):
        super(CustomDropoutLayerGetShape, self).__init__()
        self.p = p
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        if self.training:
            mask = torch.rand(x.size()) > self.p
            mask = mask.to(x.device, dtype=x.dtype)
            x = x * mask / (1 - self.p)
        return x
