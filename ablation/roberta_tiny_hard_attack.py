import time
from transformers import RobertaForSequenceClassification, BertTokenizerFast
from utils import *
from argparse import ArgumentParser

parser = ArgumentParser(description='roberta_attack')
parser.add_argument('--device', default="cuda:0", type=str, help='cuda device')
parser.add_argument('--model', default="haisongzhang/roberta-tiny-cased", type=str, help='large language model')
parser.add_argument('--dataset', default="cola", type=str, help='dataset for task')
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--lr_decay_type", default="StepLR", type=str, help="lr decay type")
parser.add_argument("--parallel", action="store_true", default=False, help="parallelize discrete optimization")
parser.add_argument("--recover_batch", default=0, type=int, help="batch to recover from")
parser.add_argument("--run", default="first", type=str, help="number of run")
args = parser.parse_args()

# This is to avoid error when running on a Mac
DEVICE = args.device if (torch.cuda.is_available() or torch.has_mps) else "cpu"
print(DEVICE)
BATCH_SIZE = args.batch_size
RESULT_FILE = f"../results/ablation/models/roberta-tiny/{args.dataset}__attack_hard_simreg_{args.run}_run_{BATCH_SIZE}.txt"
PARALLEL = args.parallel
RECOVER_BATCH = args.recover_batch
lr_decay_type = args.lr_decay_type
print("Doing parallel: ", PARALLEL)

server = RobertaForSequenceClassification.from_pretrained(args.model).to(DEVICE)
tokenizer = BertTokenizerFast.from_pretrained(args.model)
# We do not train the token embedding and position embedding layers to line up with SOTA
server.roberta.embeddings.word_embeddings.weight.requires_grad = False
server.roberta.embeddings.position_embeddings.weight.requires_grad = False

client = copy.deepcopy(server)

data_file = f"../data/{args.dataset}_data.txt"

train_sequences = []
train_labels = []

with open(data_file, "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 2):
        sequence = lines[i].strip()
        label = lines[i + 1].strip()
        train_sequences.append(sequence)
        train_labels.append(int(label))

train_data = tokenizer(train_sequences, return_tensors="pt", padding=True, truncation=True)
train_data["labels"] = torch.LongTensor(train_labels).clone()
train_data = MyDataset(train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
train_loop = tqdm(train_loader, leave=True)

batch_num = 1
for training_batch in train_loop:
    server = copy.deepcopy(client)
    client.train()
    server.train()
    loss, attentions, hidden_states, logits = train_batch_with_frozen_weights(client, training_batch, DEVICE)
    if batch_num < RECOVER_BATCH:
        batch_num += 1
        continue
    # Collect the client's gradients
    # client_name_grads = {}
    client_grads = []
    for param in client.parameters():
        if param.grad is not None:
            client_grads.append(param.grad)
    # for name, param in client.named_parameters():
    #     if param.grad is not None:
    #         client_name_grads[name] = param.grad

    # We assume the sequence length is known to line up with SOTA
    sequence_length = training_batch["input_ids"].shape[1]
    individual_lengths = []
    for i in range(training_batch["input_ids"].shape[0]):
        sequence = training_batch["input_ids"][i].tolist()
        sequence = sequence[:sequence.index(tokenizer.sep_token_id) + 1]
        individual_lengths.append(len(sequence))

    longest_length = max(individual_lengths)
    token_embedding = server.roberta.embeddings.word_embeddings.weight
    labels = training_batch["labels"].tolist()

    print("Current Reference:\n")
    for i in range(training_batch["input_ids"].shape[0]):
        print(tokenizer.decode(training_batch["input_ids"][i].tolist()))
    with open(RESULT_FILE, "a") as f:
        f.write("-------------------------\n")
        f.write(f"Batch {batch_num}:\n")
        f.write("Current Reference:\n")
        for i in range(training_batch["input_ids"].shape[0]):
            f.write(tokenizer.decode(training_batch["input_ids"][i].tolist(), skip_special_tokens=True) + "\n")

    token_set = list(range(len(tokenizer)))
    # To get the shapes of the input to all dropout layers, we create random dummy data and pass it through the model
    dummy_data = []
    for i in range(BATCH_SIZE):
        sequence = [tokenizer.cls_token_id]
        for j in range(1, longest_length - 1):
            sequence.append(random.choice(token_set))
        sequence.append(tokenizer.sep_token_id)
        sequence = sequence + [tokenizer.pad_token_id] * (sequence_length - len(sequence))
        dummy_data.append(sequence)

    # First we extract all dropout layers in the server model
    dropout_layers = []
    for name, module in server.named_modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_layers.append((name, module))


    # Create new dropout layers to get shapes of the previous layer's output
    dropout_layers_get_shape = []
    for i in range(len(dropout_layers)):
        dropout_layers_get_shape.append(CustomDropoutLayerGetShape(p=dropout_layers[i][1].p))

    # Substitute the dropout layers
    server.roberta.embeddings.dropout = dropout_layers_get_shape[0]
    j = 1
    for i in range(0, 4):
        server.roberta.encoder.layer[i].attention.self.dropout = dropout_layers_get_shape[j]
        j += 1
        server.roberta.encoder.layer[i].attention.output.dropout = dropout_layers_get_shape[j]
        j += 1
        server.roberta.encoder.layer[i].output.dropout = dropout_layers_get_shape[j]
        j += 1
    server.classifier.dropout = dropout_layers_get_shape[-1]

    # Pass the dummy data through the model to get the shapes of the input to all dropout layers
    outputs = server(input_ids=torch.LongTensor(dummy_data).to(DEVICE), labels=torch.LongTensor(labels).to(DEVICE))
    shapes = []
    for i in range(len(dropout_layers_get_shape)):
        shapes.append(dropout_layers_get_shape[i].shape)

    # Now initialise new dropout layers with fixed masks and the shapes
    dropout_layers_fixed = []
    for i in range(len(dropout_layers)):
        dropout_layer = CustomDropoutLayer(shapes[i], DEVICE, p=dropout_layers[i][1].p)
        dropout_layer.init_mask()
        dropout_layers_fixed.append(dropout_layer)

    # Substitute the dropout layers
    server.roberta.embeddings.dropout = dropout_layers_fixed[0]
    j = 1
    for i in range(0, 4):
        server.roberta.encoder.layer[i].attention.self.dropout = dropout_layers_fixed[j]
        j += 1
        server.roberta.encoder.layer[i].attention.output.dropout = dropout_layers_fixed[j]
        j += 1
        server.roberta.encoder.layer[i].output.dropout = dropout_layers_fixed[j]
        j += 1
    server.classifier.dropout = dropout_layers_fixed[-1]

    # Record the time
    start_time = time.time()
    discrete_solution = None
    discrete_solution_obj = -10000
    continuous_solution = None
    continuous_obj = -10000
    prev_discrete_solution = None

    accumulated_separate_tokens = []
    for i in range(BATCH_SIZE):
        accumulated_separate_tokens.append([])

    optimize_dropout_mask = True
    lr = 0.01
    gc.collect()
    torch.cuda.empty_cache()
    # Turn on dropout
    server.train()
    for i in range(5):
        # if i > 1:
        #     optimize_dropout_mask = False

        print("Optimize dropout mask: " + str(optimize_dropout_mask))

        continuous_optimizer = ContinuousOptimizerIndividual(copy.deepcopy(discrete_solution), copy.deepcopy(server),
                                                             tokenizer,
                                                             DEVICE, copy.deepcopy(client_grads),
                                                             copy.deepcopy(token_embedding), copy.deepcopy(labels),
                                                             alpha=0.01, beta=0,
                                                             individual_lengths=individual_lengths,
                                                             batch_size=BATCH_SIZE,
                                                             seq_len=sequence_length, init_size=2000, num_perm=2000,
                                                             lr=lr, lr_decay_type=lr_decay_type,
                                                             optimize_dropout_mask=optimize_dropout_mask,
                                                             num_of_iterations=2000, observe_embedding=True)
        server, continuous_obj, continuous_solution = continuous_optimizer.optimize()

        allocated_memory = torch.cuda.memory_allocated()
        print(f"Allocated GPU memory before: {allocated_memory / (1024 ** 3):.2f} GB")

        del continuous_optimizer
        gc.collect()
        torch.cuda.empty_cache()

        allocated_memory = torch.cuda.memory_allocated()
        print(f"Allocated GPU memory after: {allocated_memory / (1024 ** 3):.2f} GB")

        continuous_obj = calculate_discrete_obj(copy.deepcopy(continuous_solution), copy.deepcopy(labels),
                                                copy.deepcopy(server), copy.deepcopy(client_grads), DEVICE, tokenizer,
                                                alpha=0.01)
        print(f"Iter {i} continuous solution:")
        for j in range(len(continuous_solution)):
            print(tokenizer.decode(continuous_solution[j]))
        print(f"Iter {i} continuous solution obj:")
        print(continuous_obj)

        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write(f"Iter {i} continuous solution:\n")
            for j in range(len(continuous_solution)):
                f.write(tokenizer.decode(continuous_solution[j], skip_special_tokens=True) + "\n")
            f.write(f"Iter {i} continuous solution obj:\n")
            f.write(str(continuous_obj) + "\n")

        separate_tokens = []
        for solution in continuous_solution:
            separate_tokens.append(list(set(solution)))

        for i in range(BATCH_SIZE):
            accumulated_separate_tokens[i] += separate_tokens[i]
            accumulated_separate_tokens[i] = list(set(accumulated_separate_tokens[i]))

        for j in range(BATCH_SIZE):
            length = individual_lengths[j]
            # If empty, randomly picked some from tokenizers
            if not accumulated_separate_tokens[j]:
                accumulated_separate_tokens[j] = random.sample(token_set, length)

        print("Separate tokens:")
        print(accumulated_separate_tokens)

        discrete_optimizer_separate = BeamSearchOptimizerIndividual(copy.deepcopy(server), tokenizer, DEVICE,
                                                                    sequence_length,
                                                                    BATCH_SIZE, copy.deepcopy(client_grads),
                                                                    RESULT_FILE,
                                                                    copy.deepcopy(labels), [], longest_length,
                                                                    individual_lengths,
                                                                    copy.deepcopy(accumulated_separate_tokens),
                                                                    copy.deepcopy(token_set),
                                                                    parallel=PARALLEL,
                                                                    alpha=0.01,
                                                                    num_of_solutions=4,
                                                                    num_of_iter=5,
                                                                    num_of_perms=2000,
                                                                    continuous_solution=copy.deepcopy(
                                                                        continuous_solution),
                                                                    discrete_solution=copy.deepcopy(discrete_solution),
                                                                    beam=4)
        discrete_solution, discrete_solution_obj = discrete_optimizer_separate.optimize()

        allocated_memory = torch.cuda.memory_allocated()
        print(f"Allocated GPU memory before: {allocated_memory / (1024 ** 3):.2f} GB")

        del discrete_optimizer_separate
        gc.collect()
        torch.cuda.empty_cache()

        allocated_memory = torch.cuda.memory_allocated()
        print(f"Allocated GPU memory after: {allocated_memory / (1024 ** 3):.2f} GB")

        if discrete_solution == continuous_solution or discrete_solution == prev_discrete_solution:
            print("Early stopping")
            with open(RESULT_FILE, "a") as f:
                f.write("Early stopping\n")
            break
        else:
            prev_discrete_solution = discrete_solution

    if continuous_solution is not None:
        continuous_obj = calculate_discrete_obj(copy.deepcopy(continuous_solution), copy.deepcopy(labels),
                                                copy.deepcopy(server), copy.deepcopy(client_grads), DEVICE, tokenizer,
                                                alpha=0.01)

    if discrete_solution_obj > continuous_obj:
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write("Discrete solution is better\n")
            f.write(f"Discrete solution:")
            for i in range(len(discrete_solution)):
                f.write(f"{tokenizer.decode(discrete_solution[i], skip_special_tokens=True)}\n")
    else:
        with open(RESULT_FILE, "a", encoding="utf-8") as f:
            f.write("Continuous solution is better\n")
            f.write(f"Continuous solution:")
            for i in range(len(continuous_solution)):
                f.write(f"{tokenizer.decode(continuous_solution[i], skip_special_tokens=True)}\n")
    end_time = time.time()
    print("Runtime: {:.2f} seconds".format(end_time - start_time))
    with open(RESULT_FILE, "a") as f:
        f.write("Runtime: {:.2f} seconds\n".format(end_time - start_time))
    batch_num += 1
