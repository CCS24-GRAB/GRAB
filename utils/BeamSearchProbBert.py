import torch
import joblib
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def count_repetitive_n_grams(sequence, n):
    # Count how many n_grams in sequence are repetitive
    n_grams = []
    for i in range(len(sequence) - n + 1):
        n_grams.append(tuple(sequence[i:i + n]))
    count = 0
    for n_gram in n_grams:
        if n_grams.count(n_gram) > 1:
            count += 1
    return count


def count_sequences_repetitive_n_grams(previous_sequence, current_sequence, n):
    # Count how many n_grams in current sequence is already in previous sequence
    n_grams = []
    for i in range(len(current_sequence) - n + 1):
        n_grams.append(tuple(current_sequence[i:i + n]))
    previous_n_grams = []
    for i in range(len(previous_sequence) - n + 1):
        previous_n_grams.append(tuple(previous_sequence[i:i + n]))
    count = 0
    for n_gram in n_grams:
        if n_gram in previous_n_grams:
            count += 1
    return count


class BeamSearchProbBert():
    def __init__(self, client, tokenizer, token_set, batch_size, seq_len, start_token, device, generated_sentences,
                 beam_width=10):
        self.client = client
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.token_set = token_set
        self.batch_size = batch_size
        self.start_token = start_token
        self.device = device
        self.beam_width = beam_width
        self.beam_states = []
        self.beam_probs = []
        self.beam_scores = []
        if len(self.token_set) < self.beam_width:
            self.beam_width = len(self.token_set)
        self.penalty = 10
        self.generated_sentences = generated_sentences

    def search(self):
        start_sequence = [self.tokenizer.cls_token_id, self.start_token]
        with torch.no_grad():
            logits = self.client(torch.tensor([start_sequence]).to(self.device)).logits
        probs = torch.squeeze(torch.log(torch.softmax(logits, dim=-1)))
        states = []
        for token in self.token_set:
            states.append(start_sequence + [token])
        # Calculate the sequence probability
        sequence_probs = []
        for i in range(len(states)):
            sequence = states[i]
            sequence_prob = 0
            for j in range(len(sequence) - 1):
                sequence_prob += probs[j][sequence[j + 1]].item()
            sequence_probs.append(sequence_prob)
        # Calculate the score
        scores = []
        for i in range(len(states)):
            repetitive_n_grams = count_repetitive_n_grams(states[i], 2)
            score = sequence_probs[i] - self.penalty * repetitive_n_grams
            if self.generated_sentences:
                for sentence in self.generated_sentences:
                    repetitive_sequence_n_grams = count_sequences_repetitive_n_grams(sentence, states[i], 2)
                    score -= self.penalty * repetitive_sequence_n_grams
            scores.append(score)
        sorted_scores, sorted_indices = torch.sort(torch.tensor(scores), descending=True)
        # Ignore special tokens
        for i in range(self.beam_width):
            for j in range(len(sorted_scores)):
                token = states[sorted_indices[j]][-1]
                if start_sequence + [token] in self.beam_states:
                    continue
                else:
                    self.beam_scores.append(sorted_scores[j].item())
                    self.beam_states.append(start_sequence + [token])
                    self.beam_probs.append(sequence_probs[sorted_indices[j]])
                    break

        for i in range(self.seq_len - 3):
            # print(f"Searching for index {i + 3}")
            for j in range(self.beam_width):
                beam_state = self.beam_states[j]
                logits = self.client(torch.tensor([beam_state]).to(self.device)).logits
                probs = torch.squeeze(torch.log(torch.softmax(logits, dim=-1)))
                states = []
                for token in self.token_set:
                    states.append(beam_state + [token])
                sequence_probs = []
                for k in range(len(states)):
                    sequence = states[k]
                    sequence_prob = 0
                    for l in range(len(sequence) - 1):
                        sequence_prob += probs[l][sequence[l + 1]].item()
                    sequence_probs.append(sequence_prob)
                scores = []
                for k in range(len(states)):
                    repetitive_n_grams = count_repetitive_n_grams(states[k], 2)
                    score = sequence_probs[k] - self.penalty * repetitive_n_grams
                    if self.generated_sentences:
                        for sentence in self.generated_sentences:
                            repetitive_sequence_n_grams = count_sequences_repetitive_n_grams(sentence, states[k], 2)
                            score -= self.penalty * repetitive_sequence_n_grams
                    scores.append(score)
                sorted_scores, sorted_indices = torch.sort(torch.tensor(scores), descending=True)
                for k in range(len(sorted_scores)):
                    if states[sorted_indices[k]] in self.beam_states:
                        continue
                    else:
                        self.beam_scores[j] = sorted_scores[k].item()
                        self.beam_states[j] = states[sorted_indices[k]]
                        self.beam_probs[j] = sequence_probs[sorted_indices[k]]
                        break
        return self.beam_states[torch.argmax(torch.tensor(self.beam_scores))]
