import logging
import os
import sys

from datasets import load_metric

from src.utils.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from src.utils.tokenizer import Tokenization
from src.utils.dataloader import load_data
from src.custom_typing.arguments import ModelArguments, DataTrainingArguments
from transformers.trainer_utils import is_main_process
from src.utils.postprocessor import post_processing_function

# TODO: change metric
metric = load_metric('squad')


def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


def main():
    # Initializing parser
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading dataset from disk
    # TODO: Compute train_split_num automatically from number of train examples in the load_data function
    nq_dataset = load_data(data_args.dataset_path, train_split_num=25000)

    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
    )
    # Tokenize train dataset
    T = Tokenization(tokenizer)

    nq_dataset_train_tokenized = nq_dataset["train"].map(
        lambda x: T.prepare_train_features(*T.answer_mapping_train(x)),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=nq_dataset["train"].column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    nq_dataset_val_tokenized = nq_dataset["validation"].map(
        T.prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=nq_dataset["validation"].column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # loading model
    # if model checkpoint
    if "checkpoint" in model_args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer)
    )

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=nq_dataset_train_tokenized
        if training_args.do_train else None,
        eval_dataset=nq_dataset_val_tokenized if training_args.do_eval else None,
        eval_examples=nq_dataset["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    # TODO: add load_best_model to args. No option to save best model, but option to load best model after training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # TODO: Compute F1 on train dataset?


if __name__ == '__main__':
    main()
