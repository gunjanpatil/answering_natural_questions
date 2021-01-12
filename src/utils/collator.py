# custom collator for question answering
import re
import torch
import numpy as np


class Collator(object):
    """Custom collator"""

    def __init__(self, data_dict, tokenizer, max_seq_len=384, max_question_len=64):
        self.data_dict = data_dict
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_question_len = max_question_len

    def _get_all_tokens(self, candidate_words, max_answer_tokens):
        """ return candidate tokens
        Tokenize words and return the required number of tokens and
        the list of cumulative sum of number of tokens for each word at an index

        Args:
            candidate_words (List[str]): list of candidate words
            max_answer_tokens (int): max number of tokens that the answer should contain

        Returns:
            words_to_tokens_index (List[int]): cumulative sum of number of tokens for each word at an index
            candidate_tokens (List[str]): list of desired number of tokens
        """
        # cumulative sum of number of tokens for each word at an index
        words_to_tokens_index = []
        # list of all tokens, max length of the list will be equal to max_answer_tokens
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            # ignore html tags
            if re.match(r'<.+>', word):
                continue
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens) + len(tokens) > max_answer_tokens:
                break
            candidate_tokens.extend(tokens)
        return words_to_tokens_index, candidate_tokens

    def _get_positive_input_ids(self, data, question_tokens):
        max_answer_tokens = self.max_seq_len - len(question_tokens) - 3  # [CLS],[SEP],[SEP]
        candidate_start = data.positive_candidate.start_idx
        candidate_end = data.positive_candidate.end_idx
        candidate_words = data.positive_candidate.words

        words_to_tokens_index, candidate_tokens = self._get_all_tokens(candidate_words, max_answer_tokens)
        start_position, end_position = -1, -1

        if data.annotation[0]['short_answers']:
            start_position1 = data.annotation[0]['short_answers'][0]['start_token']
            end_position1 = data.annotation[0]['short_answers'][0]['end_token']
            if (start_position1 >= candidate_start and end_position1 <= candidate_end) and (
                    (end_position1 - candidate_start) < len(words_to_tokens_index)):
                start_position = words_to_tokens_index[start_position1 - candidate_start] + len(question_tokens) + 2
                end_position = words_to_tokens_index[end_position1 - candidate_start] + len(question_tokens) + 2
        return candidate_tokens, start_position, end_position

    def _get_negative_input_ids(self, data, question_tokens):
        max_answer_tokens = self.max_seq_len - len(question_tokens) - 3  # [CLS],[SEP],[SEP]
        # candidate_start = data.positive_candidate.start_idx
        # candidate_end = data.positive_candidate.end_idx
        candidate_words = data.negative_candidate.words
        _, candidate_tokens = self._get_all_tokens(candidate_words, max_answer_tokens)
        start_position, end_position = -1, -1
        return candidate_tokens, start_position, end_position

    def __call__(self, batch_ids):
        """call for generating a batch of inputs

        given a list of example ids, return batch arrays of input ids, token type ids, attention mask,
        start positions, end positions and class targets

        Args:
            batch_ids (List[int]): list of example ids

        Returns:
            batch_input_ids (Tensor(batch_size, self.max_seq_len)): token ids for positive candidate words and
                negative candidate words
            batch_attention_mask (Tensor(batch_size, self.max_seq_len)): mask for token ids which are not 0
            batch_token_type_ids (Tensor(batch_size, self.max_seq_len)): label for type of token ids whether question
                or answer
            batch_y_start (Tensor(batch_size)): starting positions for positive and negative candidates
            batch_y_end (Tensor(batch_size)): end positions for positive and negative candidate
            batch_y (Tensor(batch_size)): class labels for positive and negative candidate
        """
        # to store input for both positive and negative candidate
        batch_size = 2 * len(batch_ids)

        batch_input_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, self.max_seq_len), dtype=np.int64)

        batch_y_start = np.zeros((batch_size,), dtype=np.int64)
        batch_y_end = np.zeros((batch_size,), dtype=np.int64)
        batch_y = np.zeros((batch_size,), dtype=np.int64)

        for i, doc_id in enumerate(batch_ids):
            data = self.data_dict[doc_id]

            # get label
            annotations = data.annotation[0]
            if annotations['yes_no_answer'] == 'YES':
                batch_y[i * 2] = 4
            elif annotations['yes_no_answer'] == 'NO':
                batch_y[i * 2] = 3
            elif annotations['short_answers']:
                batch_y[i * 2] = 2
            elif annotations['long_answer']['candidate_index'] != -1:
                batch_y[i * 2] = 1
            batch_y[i * 2 + 1] = 0

            # todo: remove duplication for positive and negative candidate
            # get positive and negative samples
            question_tokens = self.tokenizer.tokenize(data.question_text)[:self.max_question_len]
            # positive
            answer_tokens, start_position, end_position = self._get_positive_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            # if annotations['short_answers']:
            #    print(data['question_text'],"[AAA]",input_tokens[start_position:end_position])
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_input_ids[i * 2, :len(input_ids)] = input_ids
            batch_token_type_ids[i * 2, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                            range(len(input_ids))]
            batch_y_start[i * 2] = start_position
            batch_y_end[i * 2] = end_position
            # negative
            answer_tokens, start_position, end_position = self._get_negative_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_token_type_ids[i * 2 + 1, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                                range(len(input_ids))]
            batch_input_ids[i * 2 + 1, :len(input_ids)] = input_ids
            batch_y_start[i * 2 + 1] = start_position
            batch_y_end[i * 2 + 1] = end_position

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(
            batch_token_type_ids), torch.LongTensor(batch_y_start), torch.LongTensor(batch_y_end), torch.LongTensor(
            batch_y)


# todo: change class name
class CollatorForHardMining(Collator):
    def __init__(self, data_dict, tokenizer, max_seq_len=384, max_question_len=64):
        super().__init__(data_dict, tokenizer, max_seq_len, max_question_len)

    def _get_input_ids(self, doc_id, candidate_index):
        data = self.data_dict[doc_id]
        question_tokens = self.tokenizer.tokenize(data['question_text'])[:self.max_question_len]
        doc_words = data['document_text'].split()

        max_answer_tokens = self.max_seq_len - len(question_tokens) - 3  # [CLS],[SEP],[SEP]
        candidate = data['long_answer_candidates'][candidate_index]
        candidate_start = candidate['start_token']
        candidate_end = candidate['end_token']
        candidate_words = doc_words[candidate_start:candidate_end]

        words_to_tokens_index, candidate_tokens = self._get_all_tokens(candidate_words, max_answer_tokens)
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + candidate_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        return input_ids, len(input_ids)

    def __call__(self, batch_ids):
        """call for generating a batch of inputs

            given a list of example ids, return batch arrays of input ids, token type ids, attention mask

            Args:
                batch_ids (List[int]): list of example ids

            Returns:
                batch_input_ids (Tensor(batch_size, self.max_seq_len)): token ids for positive candidate words and
                    negative candidate words
                batch_attention_mask (Tensor(batch_size, self.max_seq_len)): mask for token ids which are not 0
                batch_token_type_ids (Tensor(batch_size, self.max_seq_len)): label for type of token ids whether question
                    or answer
            """
        batch_size = len(batch_ids)

        batch_input_ids_temp = []
        batch_seq_len = []

        for i, (doc_id, candidate_index) in enumerate(batch_ids):
            input_ids, seq_len = self._get_input_ids(doc_id, candidate_index)
            batch_input_ids_temp.append(input_ids)
            batch_seq_len.append(seq_len)

        batch_max_seq_len = max(batch_seq_len)
        batch_input_ids = np.zeros((batch_size, batch_max_seq_len), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, batch_max_seq_len), dtype=np.int64)

        for i in range(batch_size):
            input_ids = batch_input_ids_temp[i]
            batch_input_ids[i, :len(input_ids)] = input_ids
            batch_token_type_ids[i, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                        range(len(input_ids))]

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(
            batch_token_type_ids)


class CollatorForAllExamples(Collator):
    def __init__(self, id_list, neg_id_list, data_dict, neg_data_dict, new_token_dict, tokenizer, max_seq_len=384,
                 max_question_len=64):
        super().__init__(data_dict, tokenizer, max_seq_len, max_question_len)
        self.id_list = id_list
        self.neg_id_list = neg_id_list
        self.neg_data_dict = neg_data_dict
        self.new_token_dict = new_token_dict

    def _get_all_tokens(self, candidate_words, max_answer_tokens):
        """ return candidate tokens
        Tokenize words and return the required number of tokens and
        the list of cumulative sum of number of tokens for each word at an index

        Args:
            candidate_words (List[str]): list of candidate words
            max_answer_tokens (int): max number of tokens that the answer should contain

        Returns:
            words_to_tokens_index (List[int]): cumulative sum of number of tokens for each word at an index
            candidate_tokens (List[str]): list of desired number of tokens
        """
        for i, word in enumerate(candidate_words):
            if re.match(r'<.+>', word):
                if word in self.new_token_dict:
                    candidate_words[i] = self.new_token_dict[word]
                else:
                    candidate_words[i] = '<'
        # cumulative sum of number of tokens for each word at an index
        words_to_tokens_index = []
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens) + len(tokens) > max_answer_tokens:
                break
            candidate_tokens += tokens
        return words_to_tokens_index, candidate_tokens

    def __call__(self, batch_ids):
        # 3 (doc ids) x 2 (pos and neg pair) + 2 (neg from two other doc ids) = 8 data in a batch for each process/gpu
        neg_num = 2
        batch_size = 2 * len(batch_ids) + neg_num

        batch_input_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, self.max_seq_len), dtype=np.int64)

        batch_y_start = np.zeros((batch_size,), dtype=np.int64)
        batch_y_end = np.zeros((batch_size,), dtype=np.int64)
        batch_y = np.zeros((batch_size,), dtype=np.int64)

        for i, pos_idx in enumerate(batch_ids):
            doc_id = self.id_list[pos_idx]
            data = self.data_dict[doc_id]

            # get label
            annotations = data.annotation[0]
            if annotations['yes_no_answer'] == 'YES':
                batch_y[i * 2] = 4
            elif annotations['yes_no_answer'] == 'NO':
                batch_y[i * 2] = 3
            elif annotations['short_answers']:
                batch_y[i * 2] = 2
            elif annotations['long_answer']['candidate_index'] != -1:
                batch_y[i * 2] = 1
            batch_y[i * 2 + 1] = 0

            # get positive and negative samples
            question_tokens = self.tokenizer.tokenize(data.question_text)[:self.max_question_len]
            # positive
            answer_tokens, start_position, end_position = self._get_positive_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_input_ids[i * 2, :len(input_ids)] = input_ids
            batch_token_type_ids[i * 2, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                            range(len(input_ids))]
            if annotations['short_answers']:
                if start_position < 0 or end_position < 0:  # if the groundtruth span not in the truncated data,
                    # ignore this positive data by setting labels to -1
                    batch_y_start[i * 2] = -1
                    batch_y_end[i * 2] = -1
                    batch_y[i * 2] = -1
                else:
                    batch_y_start[i * 2] = start_position
                    batch_y_end[i * 2] = end_position
            else:
                batch_y_start[i * 2] = start_position
                batch_y_end[i * 2] = end_position
            # negative
            answer_tokens, start_position, end_position = self._get_negative_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_token_type_ids[i * 2 + 1, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                                range(len(input_ids))]
            batch_input_ids[i * 2 + 1, :len(input_ids)] = input_ids
            batch_y_start[i * 2 + 1] = start_position
            batch_y_end[i * 2 + 1] = end_position

        for i, neg_idx in enumerate(batch_ids[:neg_num]):
            idx = i + 2 * len(batch_ids)
            if idx >= batch_size:
                break
            doc_id = self.neg_id_list[neg_idx]
            data = self.neg_data_dict[doc_id]

            # get negative samples
            question_tokens = self.tokenizer.tokenize(data['question_text'])[:self.max_question_len]
            # negative
            answer_tokens, start_position, end_position = self._get_negative_input_ids(data, question_tokens)
            input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + answer_tokens + ['[SEP]']
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
            batch_token_type_ids[idx, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                          range(len(input_ids))]
            batch_input_ids[idx, :len(input_ids)] = input_ids
            batch_y_start[idx] = start_position
            batch_y_end[idx] = end_position

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(
            batch_token_type_ids), torch.LongTensor(batch_y_start), torch.LongTensor(batch_y_end), torch.LongTensor(
            batch_y)
