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


class LongAnswerCandidateSimplified:
    """Class for a long answer candidate for simplified dataset"""
    def __init__(self, example: dict):
        """
        Takes in a single data line and converts it into a valid long answer candidate

        each long answer candidate will have
            example_idx : index of the example
            question_text: text of question
            annotation: full annotation for the example
            positive_candidate: correct annotation
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
        distribution[example['annotations'][0]['long_answer']['candidate_index']] = 0.
        distribution /= len(distribution)
        negative_candidate_index = sample_index(distribution)

        neg_candidate = example['long_answer_candidates'][negative_candidate_index]

        self.negative_candidate = NegativeCandidate(
            words=document_words[neg_candidate['start_token']:neg_candidate['end_token']],
            start_idx=neg_candidate['start_token'],
            end_idx=neg_candidate['end_token']
        )


class LongAnswerCandidateForHardMinedExample:
    """Class for a long answer candidate for simplified dataset"""
    def __init__(self, example: dict, distribution: dict):
        """
        Takes in a single data line and converts it into a valid long answer candidate

        each long answer candidate will have
            example_idx : index of the example
            question_text: text of question
            annotation: full annotation for the example
            positive_candidate: correct annotation
            negative_candidate: sampled negative annotation from long answer candidates
                given a probability distribution for every negative candidate

        Args:
            example: single example for simplified natural questions dataset
            distribution: probability distribution for negative long answer candidates
        """

        self.example_id = example['example_id']
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
        candidate_index_list = np.array(distribution[self.example_id]['candidate_index'])
        prob_list = np.power(np.array(distribution[self.example_id]['prob_list']), 1)
        prob_list /= sum(prob_list)
        negative_candidate_index = candidate_index_list[sample_index(prob_list)]

        neg_candidate = example['long_answer_candidates'][negative_candidate_index]

        self.negative_candidate = NegativeCandidate(
            words=document_words[neg_candidate['start_token']:neg_candidate['end_token']],
            start_idx=neg_candidate['start_token'],
            end_idx=neg_candidate['end_token']
        )


class LongAnswerNegativeCandidateForHardMinedExample:
    """Class for a long answer candidate for simplified dataset"""
    def __init__(self, example: dict, distribution: dict):
        """
        Takes in a single data line and converts it into a valid long answer candidate

        each long answer candidate will have
            example_idx : index of the example
            question_text: text of question
            annotation: full annotation for the example
            negative_candidate: sampled negative annotation from long answer candidates
                given a probability distribution for every negative candidate

        Args:
            example: single example for simplified natural questions dataset
            distribution: probability distribution for negative long answer candidates
        """

        self.example_id = example['example_id']
        self.question_text = example['question_text']
        self.annotation = example['annotations']

        document_words = example['document_text'].split()

        # sample negative candidate uniformly
        candidate_index_list = np.array(distribution[self.example_id]['candidate_index'])
        prob_list = np.power(np.array(distribution[self.example_id]['prob_list']), 1)
        prob_list /= sum(prob_list)
        negative_candidate_index = candidate_index_list[sample_index(prob_list)]

        neg_candidate = example['long_answer_candidates'][negative_candidate_index]

        self.negative_candidate = NegativeCandidate(
            words=document_words[neg_candidate['start_token']:neg_candidate['end_token']],
            start_idx=neg_candidate['start_token'],
            end_idx=neg_candidate['end_token']
        )
