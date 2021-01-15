import os
import json
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from torch import optim
import torch
import logging
import jsonlines
from apex import amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertConfig, HfArgumentParser, TrainingArguments

from src.custom_typing.candidates import sample_index, AugmentedExampleSimplified, AugmentedNegativeExampleSimplified
from custom_typing.arguments import ModelArguments, DataTrainingArguments
from src.custom_typing.datasets import SimplifiedNegativeQADataset
from src.models.bert.bert_for_qa import BertForQuestionAnswering
from src.utils.collator import CollatorV2
from src.utils.metrics import compute_loss, compute_accuracy, MovingAverage

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def get_negative_index(distribution: dict):
    """ returns a negative candidate index to sample
    Given a dictionary containing hard mined example's hardness score distribution dict, returns an index of the
    candidate to sample

    Args:
        distribution (dict): hard mined example distribution dict

    Returns:
        an integer index of the candidate to sample
    """
    candidate_index_list = np.array(distribution['candidate_index'])
    probs_list = np.power(np.array(distribution['prob_list']), 1)
    # normalizing probabilities
    probs_list /= sum(probs_list)
    return candidate_index_list[sample_index(probs_list)]


def parse_data_from_json_file(train_dataset: str, distributions_dict: dict, max_data: int = 1e10, shuffle: bool = True):
    """Reads and parses the json file for simplified natural questions dataset

    Parsing training examples

    Args:
        train_dataset (str): path to simplified training dataset jsonl file'
        distributions_dict (dict): dictionary of hard mined examples
        max_data (int): max number of examples to use for training, majorly used for debugging purposes
        shuffle (bool): set to true if training examples should be shuffled

    Returns:
        positive_ids (List[int]): list of document ids of positive examples
        negative_ids (List[int]): list of document ids of negative examples
        data_dict (dict): data dictionary of document id -> AugmentedExampleSimplified
        negative_data_dict (dict): data dictionary of document id -> AugmentedNegativeExampleSimplified
    """

    # list of document ids of positive examples (examples containing an answer)
    positive_ids = []
    # list of document ids of negative examples (examples not containing answer but have long answer candidates)
    negative_ids = []
    # data dictionary for positive examples
    data_dict = {}
    # data dictionary for negative examples
    negative_data_dict = {}

    logging.info("Parsing Training Data examples")
    with jsonlines.open(train_dataset) as reader:
        for n, data_line in enumerate(tqdm(reader)):
            if n > max_data:
                break

            is_positive = False
            annotations = data_line['annotations'][0]
            if (
                    (annotations['long_answer']['candidate_index'] != -1)
                    or annotations['short_answers']
                    or annotations['yes_no_answer'] in ['YES', 'NO']
            ):
                is_positive = True

            document_id = data_line['example_id']

            if document_id in distributions_dict:
                # sample negative candidate based on non-uniform probability distribution from hardness scores
                neg_candidate_index = get_negative_index(distributions_dict[document_id])

                # ignore positive documents with no negative long answer candidate
                # if positive example containing multiple long answer candidates
                if is_positive and len(data_line['long_answer_candidates']) > 1:
                    augmented_example = AugmentedExampleSimplified(data_line, neg_candidate_index)
                    data_dict[document_id] = augmented_example
                    positive_ids.append(document_id)

                # if not a positive example but contains long answer candidates
                elif (not is_positive) and len(data_line['long_answer_candidates']) >= 1:
                    augmented_neg_example = AugmentedNegativeExampleSimplified(data_line, neg_candidate_index)
                    negative_data_dict[document_id] = augmented_example
                    negative_ids.append(document_id)

    # length of neg_id_list must be longer than id_list otherwise data generator will error.
    # See the generator for more information.
    # todo: self added assert, why should it be true?
    assert len(negative_ids) > len(positive_ids)
    if shuffle:
        random.shuffle(positive_ids)
        random.shuffle(negative_ids)
    return positive_ids, negative_ids, data_dict, negative_data_dict


def main():
    """
    main function that loads data, model, tokenizer, config and trains the model

    Args:
        args : command line arguments
    """

    logging.info(f"local rank: {args.local_rank}")

    # Initializing hugging face parser to parse values from json file
    hf_parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = hf_parser.parse_json_file(
        json_file=os.path.abspath(args.configs)
    )

    if args.local_rank == 0:
        logging.info(f"Number of Cuda Devices: {torch.cuda.device_count}")
        logging.info("Setting up distributed training")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializing distributed training
    torch.distributed.init_process_group(backend="nccl")
    args.device = device

    # Set seed
    if args.local_rank == 0:
        logging.info(f"Setting Seed. Input Seed: {training_args.seed}")
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True

    # making sure output directory exists
    out_dir = os.path.join(data_args.project_path, training_args.output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # load hard mined examples
    with open(args.hard_mined_examples, 'rb') as f:
        distributions_dict = pickle.load(f)

    id_list, negative_id_list, data_dict, negative_data_dict = parse_data_from_json_file(args.dataset_path,
                                                                                         distributions_dict)

    if args.local_rank not in [-1, 0]:
        # blocking all processes expect base process 0
        torch.distributed.barrier()

    # load config, tokenizer and model
    config = BertConfig.from_pretrained(model_args.config_name)
    # Configuring 5 labels : {no answer, long answer, short answer, yes answer, no answer}
    config.num_labels = 5
    logging.info(f"loading tokenizer from {model_args.tokenizer_name}")
    tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, do_lower_case=True)
    logging.info(f"loading model from {model_args.model_name_or_path}")
    model = BertForQuestionAnswering.from_pretrained(model_args.model_name_or_path, config=config)

    new_token_dict = {
        '<P>': 'qw1',
        '<Table>': 'qw2',
        '<Tr>': 'qw3',
        '<Ul>': 'qw4',
        '<Ol>': 'qw5',
        '<Fl>': 'qw6',
        '<Li>': 'qw7',
        '<Dd>': 'qw8',
        '<Dt>': 'qw9',
    }
    new_token_list = list(new_token_dict.values())
    tokenizer.add_tokens(new_token_list)
    model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        # blocking base process 0 as well, thus, after this barrier is lifted for all the processes
        torch.distributed.barrier()

    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=training_args.learning_rate)
    model, optimizer = amp.initialize(model, optimizer, pt_level="O1", verbosity=0)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # training
    # iterator for training
    train_data_generator = SimplifiedNegativeQADataset(id_list, negative_id_list)
    train_sampler = DistributedSampler(train_data_generator)
    train_collator = CollatorV2(id_list=id_list,
                                neg_id_list=negative_id_list,
                                data_dict=data_dict,
                                neg_data_dict=negative_data_dict,
                                new_token_dict=new_token_dict,
                                tokenizer=tokenizer,
                                max_seq_length=data_args.max_seq_length,
                                max_question_len=data_args.max_question_length)
    train_generator = DataLoader(dataset=train_data_generator,
                                 sampler=train_sampler,
                                 collate_fn=train_collator,
                                 batch_size=training_args.per_device_train_batch_size,
                                 num_workers=3,
                                 pin_memory=True)

    # train
    start_position_loss = MovingAverage()
    end_position_loss = MovingAverage()
    classifier_loss = MovingAverage()
    start_position_accuracy = MovingAverage()
    end_position_accuracy = MovingAverage()
    classifier_accuracy = MovingAverage()

    model.train()
    best_batch_loss = float("inf")
    for epoch in tqdm(range(training_args.num_train_epochs)):
        batches_count = len(train_generator)
        if args.local_rank == 0:
            logging.info(f"training for epoch {epoch}")
            logging.info(f"Number of batches: {batches_count}")
        epoch_dir = os.path.join(out_dir, f"epoch_{epoch}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir, exist_ok=True)
        for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_target_start, batch_target_end,
                batch_target_labels) in enumerate(tqdm(train_generator)):
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            batch_target_start = batch_target_start.cuda()
            batch_target_end = batch_target_end.cuda()
            batch_target_labels = batch_target_labels.cuda()

            start_position_logits, end_position_logits, classifier_logits = model(batch_input_ids, batch_attention_mask,
                                                                                  batch_token_type_ids)
            targets = (batch_target_start, batch_target_end, batch_target_labels)
            start_loss, end_loss, class_loss = compute_loss((start_position_logits, end_position_logits,
                                                             classifier_logits), targets)
            total_batch_loss = start_loss + end_loss + class_loss
            start_acc, end_acc, class_acc, start_position_num, end_position_num, class_position_num = compute_accuracy(
                (start_position_logits, end_position_logits, classifier_logits), targets)

            start_position_loss.update(start_loss.item(), start_position_num)
            end_position_loss.update(end_loss.item(), end_position_num)
            classifier_loss.update(class_loss.item(), class_position_num)
            start_position_accuracy.update(start_acc, start_position_num)
            end_position_accuracy.update(end_acc, end_position_num)
            classifier_accuracy.update(class_acc, class_position_num)

            optimizer.zero_grad()

            if training_args.fp16:
                with amp.scale_loss(total_batch_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                total_batch_loss.backward()

            optimizer.step()

            if args.local_rank == 0:
                if (j % training_args.save_steps == 0) or j == batches_count - 1:
                    logging.info(f"epoch: {epoch},"
                                 f"training info: ,"
                                 f"start position loss: {start_position_loss.val} ,"
                                 f"end position loss: {end_position_loss.val} ,"
                                 f"classification loss: {classifier_loss.val} ,"
                                 f"start position accuracy: {start_position_accuracy.val} ,"
                                 f"end position accuracy: {end_position_accuracy.val} ,"
                                 f"classifier accuracy: {classifier_accuracy.val}")

                    logging.info(f"saving weights for step {j} to {epoch_dir} folder")
                    torch.save(model.module.state_dict(), os.path.join(epoch_dir, f"step_{j}_pytorch_model.bin"))
                if total_batch_loss < best_batch_loss:
                    torch.save(model.module.state_dict(), os.path.join(epoch_dir, "best_pytorch_model.bin"))

        if args.local_rank == 0:
            logging.info(f"epoch: {epoch},"
                         f"training info average values: ,"
                         f"start position loss: {start_position_loss.avg} ,"
                         f"end position loss: {end_position_loss.avg} ,"
                         f"classification loss: {classifier_loss.avg} ,"
                         f"start position accuracy: {start_position_accuracy.avg} ,"
                         f"end position accuracy: {end_position_accuracy.avg} ,"
                         f"classifier accuracy: {classifier_accuracy.avg}")

            logging.info(f"saving weights after epoch {epoch} to {epoch_dir} folder")
            torch.save(model.module.state_dict(), os.path.join(epoch_dir, "pytorch_model.bin"))


if __name__ == '__main__':
    # checking for system requirements
    assert torch.cuda.is_available(), "system does not have gpus to train"

    # Command line argument parser
    parser = argparse.ArgumentParser(description="arguments that can only be provided using command line")
    parser.add_argument("-r", "--local_rank", type=int, help="local gpu id provided by the torch distributed launch "
                                                             "module from command line")
    parser.add_argument("-c", "--configs", type=str, help="path to configs json file")
    parser.add_argument("-h", "--hard_mined_examples", type=str, help="path to pickle file containing hardness scores "
                                                                      "for examples")
    args = parser.parse_args()

    # make sure that the configs file is a json file
    assert os.path.splitext(args.configs)[-1] == '.json'
    assert args.hard_mined_examples, "please provide hard mined examples file to load"

    main(args)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument('-h', '--hard_mined_examples', type=str,
                        default='../outputs/hard_mined_examples/distribution_dict.pickle')
    parser.add_argument('-d', '--dataset_path', help='path to dataset examples json file', type=str,
                        default='../datasets/simplified/simplified-nq-train.jsonl')
    parser.add_argument('-m', '--model_path', help='path to a saved model', type=str, default='bert-base-uncased')
    parser.add_argument('-w', '--weights', help='path to saved weights for the model', type=str,
                        default='../weights/bert-base-uncased/epoch1/')
    parser.add_argument('-o', '--output_dir', help='path to store weights', type=str,
                        default='../weights/bert-base-uncased/finetuned/epoch0/')
    args = parser.parse_args()
    """

    main(args)
