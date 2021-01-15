"""
- Script for generating hardness scores for negative long answer candidates
- Evaluates all negative long answer candidates from simplified training dataset
  using model trained by using version 1 train script
- Requires system with CUDA installation
"""

import argparse
from collections import defaultdict

import os
import pickle
import logging
import jsonlines
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig

from src.custom_typing.datasets import SimplifiedNaturalQADataset
from src.models.bert.bert_for_qa import BertForQuestionAnswering
from src.utils.collator import CollatorForValidation

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def parse_data_from_json_file(train_dataset: str, max_data: int = 1e10):
    """Reads and parses the json file for simplified natural questions dataset

    Parsing training examples and creating a list of negative long answer candidates and a data dictionary containing
    document text, question text and long answer candidates.

    Args:
        train_dataset (str): path to simplified training dataset jsonl file
        max_data (int): max number of examples to use for parsing, majorly used for debugging purposes

    Returns:
        id_list (List[int]): list of document ids
        negative_candidates_sorted (List(document id, negative candidate index)): list of all negative long answer
            candidates sorted based on the length of the sequence (question text + long answer candidate)
        data_dict (dict): dictionary of document id -> {document text, question text, long answer candidates}

    """

    # check if the train_dataset file is of type jsonl
    assert os.path.splitext(train_dataset)[-1] == ".jsonl", "dataset file type is not jsonl, check the file provided"

    # list of document ids
    id_list = []
    # list of (document id, candidate index)
    negative_candidates = []
    # store length of all candidates
    candidates_len = []
    data_dict = {}

    logging.info("Parsing Training Data examples")
    with jsonlines.open(train_dataset) as reader:
        for n, data_line in tqdm(reader):
            if n > max_data:
                break
            document_id = data_line['example_id']
            id_list.append(document_id)

            # check if example has answer
            is_positive = False
            annotations = data_line['annotations'][0]
            if (
                    (annotations['long_answer']['candidate_index'] != -1)
                    or annotations['short_answers']
                    or annotations['yes_no_answer'] in ['YES', 'NO']
            ):
                is_positive = True

            # initializing data dict item
            data_dict[document_id] = {
                'document_text': data_line['document_text'],
                'question_text': data_line['question_text'],
                'long_answer_candidates': data_line['long_answer_candidates'],
            }

            question_len = len(data_line['question_text'].split())
            # adding negative long answer candidates
            for i in range(len(data_line['long_answer_candidates'])):
                if (is_positive and i != annotations['long_answer']['candidate_index']) \
                        or not is_positive:
                    negative_candidates.append((document_id, i))
                    candidates_len.append(question_len + data_line['long_answer_candidates'][i]['end_token'] -
                                                 data_line['long_answer_candidates'][i]['start_token'])

    # sorting candidates based on candidate's length
    sorted_index = np.argsort(np.array(candidates_len))
    negative_candidates_sorted = []
    for i in range(len(negative_candidates)):
        negative_candidates_sorted.append(negative_candidates[sorted_index[i]])

    return id_list, negative_candidates_sorted, data_dict


def main(args):
    """primary function to generate probability distributions for negative examples based on hardness

    Args:
        args : command line arguments
    """

    id_list, negative_candidates_sorted, data_dict = parse_data_from_json_file(args.dataset_path)

    # hyperparameters
    max_sequence_length = 384
    max_question_length = 64
    batch_size = 960

    # load model
    config = BertConfig.from_pretrained(args.model_path)
    config.num_labels = 5
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    model = BertForQuestionAnswering.from_pretrained(args.weights, config=config)

    model.cuda()
    if args.fp16:
        from apex import amp
        model = amp.initialize(model, opt_level=args.fp16_opt_level, verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # testing
    # iterator for testing
    test_data_generator = SimplifiedNaturalQADataset(id_list=negative_candidates_sorted)
    test_collator = CollatorForValidation(data_dict=data_dict,
                                         tokenizer=tokenizer,
                                         max_seq_length=max_sequence_length,
                                         max_question_length=max_question_length)
    test_generator = DataLoader(dataset=test_data_generator,
                                collate_fn=test_collator,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)

    # evaluating model on dataset
    logging.info('Evaluating')
    model.eval()
    # storing probability that example has an answer
    positive_probs = np.zeros((len(negative_candidates_sorted),), dtype=np.float32)
    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in tqdm(enumerate(test_generator)):
        with torch.no_grad():
            start = j * batch_size
            end = start + batch_size
            if j == len(test_generator) - 1:
                end = len(test_generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            _, _, classifier_logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            # storing probability that example has no answer
            positive_probs[start:end] += nn.functional.softmax(classifier_logits, dim=1).cpu().data.numpy()[:, 0]
    positive_probs = 1.0 - positive_probs  # storing the positive

    # initialize
    def default_val():
        return {'candidate_indices': [], 'hardness_score': []}

    distributions_dict = defaultdict(default_val)

    # from candidates to document
    for i, (document_id, candidate_index) in tqdm(enumerate(negative_candidates_sorted)):
        distributions_dict[document_id]['candidate_index_list'].append(candidate_index)
        distributions_dict[document_id]['hardness_score'].append(positive_probs[i])

    with open(args.output_path, 'wb') as f:
        pickle.dump(distributions_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    # checking for system requirements
    assert torch.cuda.is_available(), "system does not have gpus to train"

    parser = argparse.ArgumentParser(description="parser to mine hard examples from a set of examples")
    parser.add_argument('-d', '--dataset_path', help='path to dataset examples json file', type=str,
                        default='../datasets/simplified/simplified-nq-train.jsonl')
    parser.add_argument('-o', '--output_path', help='path to store hard mined examples', type=str,
                        default='../outputs/hard_mined_examples/distribution_dict.pickle')
    parser.add_argument('-m', '--model_path', help='path to a saved model', type=str, default='bert-base-uncased')
    parser.add_argument('-w', '--weights', help='path to saved weights for the model', type=str,
                        default='../weights/bert-base-uncased/epoch1/')
    parser.add_argument('--fp16', action='store_true', help='mention if loaded model is trained on half precision')
    parser.add_argument('--fp16_opt_level', default='O1', type=int, help='mention the opt level for mixed precision')
    args = parser.parse_args()

    main(args)
