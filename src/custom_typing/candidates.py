"""
Custom template for augmenting examples from simplified dataset
"""

from typing import Optional, List
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Candidate:
    """candidate text, start, and end"""
    words: Optional[List[str]] = field(default=None, metadata={
        "help": "List of words for the candidate text"
    })

    start_idx: Optional[int] = field(default=None, metadata={
        "help": "Start index for the candidate"
    })

    end_idx: Optional[int] = field(default=None, metadata={
        "help": "End index for the candidate"
    })


@dataclass
class NegativeCandidate(Candidate):
    """Negative candidate inherited from candidate"""
    pass


@dataclass
class PositiveCandidate(Candidate):
    """Positive candidate inherited from candidate"""
    pass


def sample_index(distribution: List[float]):
    """Random sample
    randomly sample an index from a given distribution

    Args:
        distribution (List[float]: list of probabilities to pick an index

    Returns:
        index (int)
    """
    temp = np.random.random()
    value = 0.
    for index in range(len(distribution)):
        value += distribution[index]
        if value > temp:
            break
    return index


class AugmentedExampleSimplified:
    """Class to create an augmented positive example containing positive long answer candidate and
    negative long answer candidate for simplified train dataset"""

    def __init__(self, example: dict):
        """
        Takes in a single positive example and converts it into a new augmented type example

        Assumes that the example contains a positive long answer candidate and negative long answer candidates

        each long answer candidate will have
            example_idx : index of the example
            question_text: text of question
            annotation: full annotation for the example
            positive_candidate: correct long answer candidate/annotation
            negative_candidate: randomly sampled wrong annotation from long answer candidates

        Args:
            example: single example for simplified natural questions dataset
        """

        self.example_idx = example['example_id']
        self.question_text = example['question_text']
        self.annotation = example['annotations']

        document_words = example['document_text'].split()
        pos_candidate = example['long_answer_candidates'][example['annotations'][0]['long_answer']['candidate_index']]

        self.positive_candidate = PositiveCandidate(
            words=document_words[pos_candidate['start_token']:pos_candidate['end_token']],
            start_idx=pos_candidate['start_token'],
            end_idx=pos_candidate['end_token']
        )

        # sample negative candidate uniformly
        distribution = np.ones((len(example['long_answer_candidates']),), dtype=np.float32)
        distribution[example['annotations'][0]['long_answer']['candidate_index']] = 0.0
        distribution /= len(distribution)
        negative_candidate_index = sample_index(distribution)

        neg_candidate = example['long_answer_candidates'][negative_candidate_index]

        self.negative_candidate = NegativeCandidate(
            words=document_words[neg_candidate['start_token']:neg_candidate['end_token']],
            start_idx=neg_candidate['start_token'],
            end_idx=neg_candidate['end_token']
        )
