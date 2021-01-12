"""
Training model with google's simplified version of natural questions dataset
"""
import os
import random
import argparse
import jsonlines

import numpy as np
import torch as torch

from apex import amp
from torch.utils.data import DistributedSampler, DataLoader

from tqdm import tqdm
import logging
from transformers import BertTokenizer, BertConfig, HfArgumentParser, TrainingArguments

from custom_typing.arguments import ModelArguments, DataTrainingArguments
from custom_typing.candidates import LongAnswerCandidateSimplified
from custom_typing.datasets import SimplifiedNaturalQADataset
from models.bert.bert_for_qa import BertForQuestionAnswering
from utils.collator import Collator
from utils.metrics import MovingAverage, compute_loss, compute_accuracy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


def parse_data_from_json_file(json_file: str, *, max_data: int = 1e10, shuffle: bool = True):
    """Reads and parses the json file for simplified natural questions dataset"""
    id_list = []
    data_dict = {}

    logging.info("Parsing Training Data examples")
    with jsonlines.open(json_file) as reader:
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

            if is_positive and len(data_line['long_answer_candidates']) > 1:
                long_answer_candidate = LongAnswerCandidateSimplified(example=data_line)
                data_dict[long_answer_candidate.example_idx] = long_answer_candidate
                id_list.append(long_answer_candidate.example_idx)

    if shuffle:
        random.shuffle(id_list)
    logging.info(f"Number of documents in training set: {len(id_list)}")
    return id_list, data_dict


def main():

    # checking for system requirements
    assert torch.cuda.is_available(), "system does not have multiple gpus to train"
    assert torch.cuda.device_count() > 1, "system does not have mutliple gpus to train"

    # Command line argument parser
    parser = argparse.ArgumentParser(description="arguments that can only be provided using command line")
    parser.add_argument("-r", "--local_rank", type=int, help="local gpu id provided by the torch distributed launch "
                                                             "module from command line")
    parser.add_argument("-c", "--configs", type=str, help="path to configs json file")
    args = parser.parse_args()
    logging.info(f"local rank: {args.local_rank}")

    # Initializing hugging face parser to parse values from json file
    hf_parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    # make sure that the first argument is a json file to configs
    assert os.path.splitext(args.configs)[-1] == '.json'
    model_args, data_args, training_args = hf_parser.parse_json_file(
        json_file=os.path.abspath(args.configs)
    )

    num_cuda_devices = torch.cuda.device_count()
    if args.local_rank == 0:
        logging.info(f"Number of Cuda Devices: {num_cuda_devices}")
        logging.info("Setting up distributed training")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
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

    out_dir = os.path.join(data_args.project_path, training_args.output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    dataset_train_file = os.path.join(data_args.datasets_path, data_args.simplified_train_dataset)
    idx_list, data_dict = parse_data_from_json_file(dataset_train_file)

    if args.local_rank not in [-1, 0]:
        # blocking all processes expect base process 0
        torch.distributed.barrier()

    # load config, tokenizer and model
    logging.info(f"loading config from {model_args.config_name}")
    config = BertConfig.from_pretrained(model_args.config_name)
    config.num_labels = 5
    logging.info(f"loading tokenizer from {model_args.tokenizer_name}")
    tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, do_lower_case=True)
    logging.info(f"loading model from {model_args.model_name_or_path}")
    model = BertForQuestionAnswering.from_pretrained(model_args.model_name_or_path, config=config)

    if args.local_rank == 0:
        # blocking base process 0 as well, thus, after this barrier is lifted for all the processes
        torch.distributed.barrier()

    model.to(device)

    # Initialize optimizer
    logging.info(f"initializing Adam optimizer with learning rate {training_args.learning_rate}")
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    if training_args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=training_args.fp16_opt_level, verbosity=0)
    logging.info("distributing model paralelly")
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # iterator for training
    train_data_generator = SimplifiedNaturalQADataset(id_list=idx_list)
    train_sampler = DistributedSampler(train_data_generator, rank=args.local_rank)
    train_collate = Collator(data_dict=data_dict,
                             tokenizer=tokenizer,
                             max_seq_len=data_args.max_seq_length,
                             max_question_len=data_args.max_question_length)
    train_generator = DataLoader(dataset=train_data_generator,
                                 sampler=train_sampler,
                                 collate_fn=train_collate,
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
        if args.local_rank == 0:
            logging.info(f"training for epoch {epoch}")
            batches_count = len(train_generator)
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
                if (j % training_args.save_steps == 0) or j == batches_count-1:
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
            torch.save(model.module.state_dict(), os.path.join(epoch_dir, "final_pytorch_model.bin"))


if __name__ == '__main__':
    main()
