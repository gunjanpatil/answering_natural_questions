# coding=utf-8
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to directory to store the pretrained models downloaded from huggingface.co"
        },
    )
    weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pytorch_model.bin file. If None, will be same as model_name_or_path"
        }
    )

    def __post_init__(self):
        if self.weights is None:
            self.weights = self.model_name_or_path
        if self.config_name is None:
            self.config_name = self.model_name_or_path
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    datasets_path: Optional[str] = field(
        default="/home/gunjan/projects/datasets",
        metadata={
            "help": "absolute path to datasets dir"
        }
    )
    project_path: Optional[str] = field(
        default="/home/gunjan/projects/answering_natural_questions",
        metadata={
            "help": "absolute path to project dir"
        }
    )
    simplified_train_dataset: Optional[str] = field(
        default="natural_questions_simplified/v1.0-simplified_simplified-nq-train.jsonl",
        metadata={
            "help": "relative path to simplified dataset"
        }
    )
    simplified_dev_dataset: Optional[str] = field(
        default="natural_questions_simplified/v1.0-simplified_simplified-nq-train.jsonl",
        metadata={
            "help": "relative path to simplified dataset"
        }
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_question_length: int = field(
        default=64,
        metadata={
            "help": "The maximum question length after tokenization. questions longer "
                    "than this will be truncated, question shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if self.datasets_path is None:
            raise ValueError(
                "Need path to datasets folder"
            )
        elif self.simplified_train_dataset is None:
            raise ValueError(
                "Need simplified training dataset file name"
            )
        else:
            if self.simplified_train_dataset is not None:
                extension = os.path.splitext(self.simplified_train_dataset)[-1]
                assert extension == ".jsonl", "train file should be a jsonl file."

