# custom collator for question answering
import re
import torch
import numpy as np


class Collator(object):
    """Custom collator for simplified training dataset"""

    def __init__(self, data_dict, tokenizer, max_seq_length=384, max_question_length=64):
        """
        Initializes object member variables

        Args:
            data_dict (dict): dictionary of training examples: document id: data
            tokenizer (BertTokenizer): pretrained tokenizer from hugging face to tokenize input words
            max_seq_length (int): maximum length(words) of a training example
            max_question_length (int): maximum length(words) of a question
        """
        self.data_dict = data_dict
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_question_len = max_question_length

    def _get_all_tokens(self, candidate_words, answer_tokens_length):
        """ returns words to token indices and candidate tokens

        Tokenize words and return the required number of tokens and
        the list of cumulative sum of number of tokens for each word at an index

        Args:
            candidate_words (List[str]): list of candidate words
            answer_tokens_length (int): max number of tokens that the answer should contain

        Returns:
            words_to_tokens_index (List[int]): list of token indices for words
            candidate_tokens (List[str]): list of desired number of tokens (max length = max_answer_tokens)
        """
        # list of token indices for words
        words_to_tokens_index = []
        # list of all tokens, max length of the list will be equal to max_answer_tokens
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            # ignore html tags
            if re.match(r'<.+>', word):
                continue
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens) + len(tokens) > answer_tokens_length:
                break
            candidate_tokens.extend(tokens)
        return words_to_tokens_index, candidate_tokens

    def _get_positive_answer_tokens(self, data, question_tokens_length, answer_tokens_length):
        """returns candidate tokens, start position and end position for positive example

        Given data and length of question tokens, returns candidate tokens, start position from short answer,
        end position from short answer for positive example

        Args:
            data (AugmentedExampleSimplified): augmented example for a particular document id, contains annotations,
                question text, positive candidate, negative candidate, document text, ..
            question_tokens_length (int): number of tokens in the question
            answer_tokens_length (int): number of tokens in the answer

        Returns:
            candidate_tokens(List[str]): list of size=answer_tokens_length that contains tokens for positive
                candidate words
            start_position (int): token index corresponding to the start token of short answer
            end_position (int): token index corresponding to the end token of short answer

        """
        candidate_start = data.positive_candidate.start_idx
        candidate_end = data.positive_candidate.end_idx
        candidate_words = data.positive_candidate.words

        words_to_tokens_index, candidate_tokens = self._get_all_tokens(candidate_words, answer_tokens_length)
        start_position, end_position = -1, -1

        if data.annotation[0]['short_answers']:
            start_position1 = data.annotation[0]['short_answers'][0]['start_token']
            end_position1 = data.annotation[0]['short_answers'][0]['end_token']
            if (start_position1 >= candidate_start and end_position1 <= candidate_end) and (
                    (end_position1 - candidate_start) < len(words_to_tokens_index)):
                start_position = words_to_tokens_index[start_position1 - candidate_start] + question_tokens_length + 2
                end_position = words_to_tokens_index[end_position1 - candidate_start] + question_tokens_length + 2
        return candidate_tokens, start_position, end_position

    def _get_negative_answer_tokens(self, data, answer_tokens_length):
        """returns candidate tokens, start position and end position for negative example

        Given data and length of question tokens, returns candidate tokens, start position from short answer,
        end position from short answer for positive example

        Args:
            data (AugmentedExampleSimplified): augmented example for a particular document id, contains annotations,
                question text, positive candidate, negative candidate, document text, ..
            question_tokens_length (int): number of tokens in the question
            answer_tokens_length (int): number of tokens in the answer

        Returns:
            candidate_tokens(List[str]): list of size=answer_tokens_length that contains tokens for negative
                candidate words
            start_position (int): token index corresponding to the start token of short answer, by default -1
            end_position (int): token index corresponding to the end token of short answer, by default -1

        """
        candidate_words = data.negative_candidate.words
        _, candidate_tokens = self._get_all_tokens(candidate_words, answer_tokens_length)
        start_position, end_position = -1, -1
        return candidate_tokens, start_position, end_position

    def __call__(self, batch_ids):
        """call for generating a batch of inputs

        given a list of example ids, return batch tensors of input ids, token type ids, attention mask,
        start positions, end positions and class labels

        Args:
            batch_ids (List[int]): list of example ids

        Returns:
            batch_input_ids (Tensor(batch_size, self.max_seq_len)): token ids for positive candidate words and
                negative candidate words
            batch_attention_mask (Tensor(batch_size, self.max_seq_len)): mask for token ids which are not 0
            batch_token_type_ids (Tensor(batch_size, self.max_seq_len)): label for type of token ids whether question
                or answer
            batch_start_indices (Tensor(batch_size)): starting positions for positive and negative candidates
            batch_end_indices (Tensor(batch_size)): end positions for positive and negative candidate
            batch_class_labels (Tensor(batch_size)): class labels for positive and negative candidate
        """
        # to store input for both positive and negative candidate
        batch_size = 2 * len(batch_ids)

        # initializing batch inputs
        batch_input_ids = np.zeros((batch_size, self.max_seq_length), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, self.max_seq_length), dtype=np.int64)
        batch_start_indices = np.zeros((batch_size,), dtype=np.int64)
        batch_end_indices = np.zeros((batch_size,), dtype=np.int64)
        batch_class_labels = np.zeros((batch_size,), dtype=np.int64)

        for i, doc_id in enumerate(batch_ids):
            data = self.data_dict[doc_id]

            # get label
            annotations = data.annotation[0]
            if annotations['yes_no_answer'] == 'YES':
                batch_class_labels[i * 2] = 4
            elif annotations['yes_no_answer'] == 'NO':
                batch_class_labels[i * 2] = 3
            elif annotations['short_answers']:
                batch_class_labels[i * 2] = 2
            elif annotations['long_answer']['candidate_index'] != -1:
                batch_class_labels[i * 2] = 1
            batch_class_labels[i * 2 + 1] = 0

            question_tokens = self.tokenizer.tokenize(data.question_text)[:self.max_question_len]
            answer_tokens_length = self.max_seq_length - len(question_tokens) - 3

            # get positive example using the positive candidate
            positive_answer_tokens, positive_start_index, positive_end_index = \
                self._get_positive_answer_tokens(data, len(question_tokens), answer_tokens_length)
            positive_input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + positive_answer_tokens + ['[SEP]']
            positive_input_ids = self.tokenizer.convert_tokens_to_ids(positive_input_tokens)
            batch_input_ids[i * 2, :len(positive_input_ids)] = positive_input_ids
            batch_token_type_ids[i * 2, :len(positive_input_ids)] = [0 if k <= positive_input_ids.index(102) else 1
                                                                     for k in range(len(positive_input_ids))]
            batch_start_indices[i * 2] = positive_start_index
            batch_end_indices[i * 2] = positive_end_index

            # freeing memory
            del positive_answer_tokens, positive_input_tokens, positive_input_ids

            # get negative example from negative candidate
            negative_answer_tokens, negative_start_index, negative_end_index = \
                self._get_negative_answer_tokens(data, answer_tokens_length)
            negative_input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + negative_answer_tokens + ['[SEP]']
            negative_input_ids = self.tokenizer.convert_tokens_to_ids(negative_input_tokens)
            batch_token_type_ids[i * 2 + 1, :len(negative_input_ids)] = [0 if k <= negative_input_ids.index(102) else 1
                                                                         for k in range(len(negative_input_ids))]
            batch_input_ids[i * 2 + 1, :len(negative_input_ids)] = negative_input_ids
            batch_start_indices[i * 2 + 1] = negative_start_index
            batch_end_indices[i * 2 + 1] = negative_end_index

            # freeing memory
            del negative_answer_tokens, negative_input_tokens, negative_input_ids

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(
            batch_token_type_ids), torch.LongTensor(batch_start_indices), torch.LongTensor(batch_end_indices), torch.LongTensor(
            batch_class_labels)


class CollatorForValidation(Collator):
    """
    custom collator for validating only long answer type questions
    """
    def __init__(self, data_dict, tokenizer, max_seq_length=384, max_question_length=64):
        """
        Initializes object member variables

        Args:
            data_dict (dict): dictionary of training examples: document id: data
            tokenizer (BertTokenizer): pretrained tokenizer from hugging face to tokenize input words
            max_seq_length (int): maximum length(words) of a training example
            max_question_length (int): maximum length(words) of a question
        """
        super().__init__(data_dict, tokenizer, max_seq_length, max_question_length)

    def _get_input_ids(self, doc_id, candidate_index):
        """
        given document id and long answer candidate index, returns example input ids, words to tokens index

        Args:
            doc_id (int): document id
            candidate_index (int): long answer candidate index

        Returns:
            input_ids List[int]: list of tokens ids for input sequence ([CLS]+question+[SEP]+answer+[SEP])
            words_to_tokens_index (List[int]): list of token indices for words

        """
        data = self.data_dict[doc_id]
        question_tokens = self.tokenizer.tokenize(data['question_text'])[:self.max_question_len]
        doc_words = data['document_text'].split()

        answer_tokens_length = self.max_seq_length - len(question_tokens) - 3  # [CLS],[SEP],[SEP]
        candidate = data['long_answer_candidates'][candidate_index]
        candidate_words = doc_words[candidate['start_token']:candidate['end_token']]

        words_to_tokens_index, candidate_tokens = self._get_all_tokens(candidate_words, answer_tokens_length)
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + candidate_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        return input_ids


    def __call__(self, batch_ids):
        """call for generating a batch of inputs from simplified validation dataset

        given a list of example ids, return batch tensors of input ids, token type ids, attention mask,
        words to token index, batch offset, batch max seq length

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

        # initializing batch inputs
        batch_input_ids_temp = []
        batch_max_seq_length = 0

        for i, (doc_id, candidate_index) in enumerate(batch_ids):
            input_ids = self._get_input_ids(doc_id, candidate_index)
            batch_input_ids_temp.append(input_ids)
            batch_max_seq_length = max(len(input_ids), batch_max_seq_length)

        batch_input_ids = np.zeros((batch_size, batch_max_seq_length), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, batch_max_seq_length), dtype=np.int64)

        for i in range(batch_size):
            input_ids = batch_input_ids_temp[i]
            batch_input_ids[i, :len(input_ids)] = input_ids
            batch_token_type_ids[i, :len(input_ids)] = [0 if k <= input_ids.index(102) else 1 for k in
                                                        range(len(input_ids))]

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(
            batch_token_type_ids)


class CollatorV2(Collator):
    """Custom collator for simplified training dataset and hard mined examples"""

    def __init__(self, id_list, neg_id_list, data_dict, neg_data_dict, new_token_dict, tokenizer, max_seq_length=384,
                 max_question_length=64):
        super().__init__(data_dict, tokenizer, max_seq_length, max_question_length)
        self.id_list = id_list
        self.neg_id_list = neg_id_list
        self.neg_data_dict = neg_data_dict
        self.new_token_dict = new_token_dict

    def _get_all_tokens(self, candidate_words, answer_tokens_length):
        """ returns words to token indices and candidate tokens

        Tokenize words and return the required number of tokens and
        the list of cumulative sum of number of tokens for each word at an index

        Args:
            candidate_words (List[str]): list of candidate words
            answer_tokens_length (int): max number of tokens that the answer should contain

        Returns:
            words_to_tokens_index (List[int]): list of token indices for words
            candidate_tokens (List[str]): list of desired number of tokens (max length = max_answer_tokens)
        """
        for i, word in enumerate(candidate_words):
            if re.match(r'<.+>', word):
                if word in self.new_token_dict:
                    candidate_words[i] = self.new_token_dict[word]
                else:
                    candidate_words[i] = '<'
        # list of token indices for words
        words_to_tokens_index = []
        # list of all tokens, max length of the list will be equal to max_answer_tokens
        candidate_tokens = []
        for i, word in enumerate(candidate_words):
            words_to_tokens_index.append(len(candidate_tokens))
            # ignore html tags
            if re.match(r'<.+>', word):
                continue
            tokens = self.tokenizer.tokenize(word)
            if len(candidate_tokens) + len(tokens) > answer_tokens_length:
                break
            candidate_tokens.extend(tokens)
        return words_to_tokens_index, candidate_tokens

    def __call__(self, batch_ids):
        """call for generating a batch of inputs

        Args:
            batch_ids (List[int]): list of example ids

        Returns:
            batch_input_ids (Tensor(batch_size, self.max_seq_len)): token ids for positive candidate words and
                negative candidate words
            batch_attention_mask (Tensor(batch_size, self.max_seq_len)): mask for token ids which are not 0
            batch_token_type_ids (Tensor(batch_size, self.max_seq_len)): label for type of token ids whether question
                or answer
            batch_start_indices (Tensor(batch_size)): starting positions for positive and negative candidates
            batch_end_indices (Tensor(batch_size)): end positions for positive and negative candidate
            batch_class_labels (Tensor(batch_size)): class labels for positive and negative candidate
        """
        # defining batch size value
        negative_examples_count = 2
        batch_size = 2 * len(batch_ids) + negative_examples_count  # two negative sample per batch

        # initializing batch inputs
        batch_input_ids = np.zeros((batch_size, self.max_seq_length), dtype=np.int64)
        batch_token_type_ids = np.ones((batch_size, self.max_seq_length), dtype=np.int64)
        batch_start_indices = np.zeros((batch_size,), dtype=np.int64)
        batch_end_indices = np.zeros((batch_size,), dtype=np.int64)
        batch_class_labels = np.zeros((batch_size,), dtype=np.int64)

        for i, pos_id in enumerate(batch_ids):
            document_id = self.id_list[pos_id]
            data = self.data_dict[document_id]

            # get label
            annotations = data.annotation[0]
            if annotations['yes_no_answer'] == 'YES':
                batch_class_labels[i * 2] = 4
            elif annotations['yes_no_answer'] == 'NO':
                batch_class_labels[i * 2] = 3
            elif annotations['short_answers']:
                batch_class_labels[i * 2] = 2
            elif annotations['long_answer']['candidate_index'] != -1:
                batch_class_labels[i * 2] = 1
            batch_class_labels[i * 2 + 1] = 0

            question_tokens = self.tokenizer.tokenize(data.question_text)[:self.max_question_len]
            answer_tokens_length = self.max_seq_length - len(question_tokens) - 3

            # get positive example using the positive candidate
            positive_answer_tokens, positive_start_index, positive_end_index = \
                self._get_positive_answer_tokens(data, len(question_tokens), answer_tokens_length)
            positive_input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + positive_answer_tokens + ['[SEP]']
            positive_input_ids = self.tokenizer.convert_tokens_to_ids(positive_input_tokens)
            batch_input_ids[i * 2, :len(positive_input_ids)] = positive_input_ids
            batch_token_type_ids[i * 2, :len(positive_input_ids)] = [0 if k <= positive_input_ids.index(102) else 1
                                                                     for k in range(len(positive_input_ids))]
            if annotations['short_answers']:
                if positive_start_index < 0 or positive_end_index < 0:  # if the groundtruth span not in the truncated data,
                    # ignore this positive data by setting labels to -1
                    batch_start_indices[i * 2] = -1
                    batch_end_indices[i * 2] = -1
                    batch_class_labels[i * 2] = -1
                else:
                    batch_start_indices[i * 2] = positive_start_index
                    batch_end_indices[i * 2] = positive_end_index
            else:
                batch_start_indices[i * 2] = positive_start_index
                batch_end_indices[i * 2] = positive_end_index

            # freeing memory
            del positive_answer_tokens, positive_input_tokens, positive_input_ids

            # get negative example from negative candidate
            negative_answer_tokens, negative_start_index, negative_end_index = \
                self._get_negative_answer_tokens(data, answer_tokens_length)
            negative_input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + negative_answer_tokens + ['[SEP]']
            negative_input_ids = self.tokenizer.convert_tokens_to_ids(negative_input_tokens)
            batch_token_type_ids[i * 2 + 1, :len(negative_input_ids)] = [0 if k <= negative_input_ids.index(102) else 1
                                                                         for k in range(len(negative_input_ids))]
            batch_input_ids[i * 2 + 1, :len(negative_input_ids)] = negative_input_ids
            batch_start_indices[i * 2 + 1] = negative_start_index
            batch_end_indices[i * 2 + 1] = negative_end_index

            # freeing memory
            del negative_answer_tokens, negative_input_tokens, negative_input_ids

        for i, neg_id in enumerate(batch_ids[:negative_examples_count]):
            id = i + 2 * len(batch_ids)
            if id >= batch_size:
                break
            document_id = self.neg_id_list[neg_id]
            data = self.neg_data_dict[document_id]

            question_tokens = self.tokenizer.tokenize(data.question_text)[:self.max_question_len]
            answer_tokens_length = self.max_seq_length - len(question_tokens) - 3
            # get negative example from negative candidate
            negative_answer_tokens, negative_start_index, negative_end_index = \
                self._get_negative_answer_tokens(data, answer_tokens_length)
            negative_input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + negative_answer_tokens + ['[SEP]']
            negative_input_ids = self.tokenizer.convert_tokens_to_ids(negative_input_tokens)
            batch_token_type_ids[i * 2 + 1, :len(negative_input_ids)] = [0 if k <= negative_input_ids.index(102) else 1
                                                                         for k in range(len(negative_input_ids))]
            batch_input_ids[i * 2 + 1, :len(negative_input_ids)] = negative_input_ids
            batch_start_indices[i * 2 + 1] = negative_start_index
            batch_end_indices[i * 2 + 1] = negative_end_index

            # freeing memory
            del negative_answer_tokens, negative_input_tokens, negative_input_ids

        batch_attention_mask = batch_input_ids > 0

        return torch.from_numpy(batch_input_ids), torch.from_numpy(batch_attention_mask), torch.from_numpy(
            batch_token_type_ids), torch.LongTensor(batch_start_indices), torch.LongTensor(batch_end_indices), torch.LongTensor(
            batch_class_labels)
