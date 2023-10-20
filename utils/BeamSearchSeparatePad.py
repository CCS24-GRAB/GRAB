from .GreedyOptimizerSeparatePerm import GreedyOptimizerSeparatePad
import copy


class BeamSearchSeparatePad(GreedyOptimizerSeparatePad):
    def __init__(self, client, server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                 labels, longest_length, separate_tokens, alpha=0.5, num_of_iter=5, prev_solution=None, beam_size=5):
        super().__init__(client, server, tokenizer, device, seq_len, batch_size, client_grads, result_file,
                         labels, longest_length, separate_tokens, alpha, num_of_iter, prev_solution)
        self.beam_size = beam_size
        self.beams = []
        self.beam_objs = []

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

    def optimize_init(self):
        separate_tokens = self.separate_tokens[0]
        non_special_tokens = [token for token in separate_tokens if
                              token not in self.tokenizer.all_special_ids]
        solutions = []
        objs = []
        for token in non_special_tokens:
            solution = copy.deepcopy(self.solution)
            sequence = solution[0]
            sequence[1] = token
            solution[0] = sequence
            obj_value = self.calculate_obj_value(solution)
            solutions.append(solution)
            objs.append(obj_value)

        # Sort the solutions and objs
        sorted_objs, sorted_solutions = zip(*sorted(zip(objs, solutions), reverse=True))
        self.beams = list(sorted_solutions[:self.beam_size])
        self.beam_objs = list(sorted_objs[:self.beam_size])



        # for l in range(len(self.beams)):
        #     for i in range(self.batch_size):
        #         separate_tokens = self.separate_tokens[i]
        #         non_special_tokens = [token for token in separate_tokens if
        #                               token not in self.tokenizer.all_special_ids]
        #         for j in range(2, self.longest_length):
        #             solutions = []
        #             objs = []
        #             for token in non_special_tokens:
        #                 solution = copy.deepcopy(self.beams[l])
        #                 sequence = solution[i]
        #                 sequence[j] = token
        #                 solution[i] = sequence
        #                 obj_value = self.calculate_obj_value(solution)
        #                 solutions.append(solution)
        #                 objs.append(obj_value)



