from .GreedyOptimizerIndividual import GreedyOptimizerIndividual
import copy
import numpy as np


class BeamSearchOptimizerIndividualLongest(GreedyOptimizerIndividual):
    def __init__(self, server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                 labels, start_tokens, longest_length, individual_lengths, separate_tokens, token_set, parallel,
                 alpha=0.05,
                 num_of_solutions=64, num_of_iter=5, num_of_perms=2000, continuous_solution=None,
                 discrete_solution=None, beam=5):
        super().__init__(server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                         labels, start_tokens, longest_length, individual_lengths, separate_tokens, token_set, parallel,
                         alpha,
                         num_of_solutions, num_of_iter, num_of_perms, continuous_solution, discrete_solution)
        self.beam = beam
        for tokens in self.separate_tokens:
            tks = [tk for tk in tokens if tk != self.tokenizer.cls_token_id]
            if len(tks) < self.beam:
                self.beam = len(tks)

    def optimize_one_solution(self, index):
        print(f"optimizing solution {index}")
        current_best_obj = self.obj_values[index]
        current_best_solution = copy.deepcopy(self.solutions[index])
        current_beams = []
        current_beams_obj = []
        # if self.start_tokens == self.non_special_token_set:
        #     start = 1
        # else:
        #     start = 2
        for j in range(self.batch_size):
            current_solution = copy.deepcopy(current_best_solution)
            separate_tokens = self.separate_tokens[j]
            length = self.individual_lengths[j]
            for k in range(1, length - 1):
                temp_solutions = []
                temp_objs = []
                for token in [tk for tk in separate_tokens if tk != self.tokenizer.cls_token_id]:
                    if not current_beams:
                        sequence = copy.deepcopy(current_solution[j])
                        sequence[k] = token
                        temp_solution = copy.deepcopy(current_solution)
                        temp_solution[j] = sequence
                        temp_obj = self.calculate_obj_value(temp_solution)
                        temp_solutions.append(temp_solution)
                        temp_objs.append(temp_obj)
                    else:
                        for beam in current_beams:
                            sequence = copy.deepcopy(beam[j])
                            sequence[k] = token
                            temp_solution = copy.deepcopy(beam)
                            temp_solution[j] = sequence
                            temp_obj = self.calculate_obj_value(temp_solution)
                            temp_solutions.append(temp_solution)
                            temp_objs.append(temp_obj)
                beam_indexes = (np.argpartition(temp_objs, -self.beam)[-self.beam:]).tolist()
                current_beams = []
                current_beams_obj = []
                for index in beam_indexes:
                    current_beams.append(copy.deepcopy(temp_solutions[index]))
                    current_beams_obj.append(copy.deepcopy(temp_objs[index]))
        if np.max(current_beams_obj) > current_best_obj:
            current_best_solution = copy.deepcopy(current_beams[np.argmax(current_beams_obj)])
        return current_best_solution
