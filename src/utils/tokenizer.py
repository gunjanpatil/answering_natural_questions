"""
Prepare / Generate tokenized dataset
"""
from datasets import DatasetDict
from typing import List, Dict, Any


class Tokenization:
    """
    Tokenizing dataset helper functions
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def single_answer_mapping_train(document: dict, long_answer: list) -> dict:
        """Generate answer for a single example

        Generate answer containing answer_start index and text for each example

        Args:
            document (dict): dictionary containing document information
            long_answer (list): list containing single long answer dictionary
                item for training example

        Returns:
            dict containing answer_start -> list[start idx of answer] and
                text -> str (answer text)
        """
        start_token = long_answer[0]["start_token"]
        end_token = long_answer[0]["end_token"]
        answer_start = 0
        for token in document["tokens"]["token"][:start_token]:
            answer_start += len(token) + 1
        return {
            "answer_start": [answer_start],
            "text": [" ".join(document["tokens"]["token"][start_token:end_token])],
        }

    def answer_mapping_train(self, examples: dict):
        """
        Generating a list of contexts, questions, and answers from
        a list of natural questions train examples.

        Args:
            examples (dict): dictionary of natural questions train examples,
                contains id, questions, document, annotations as keys

        Returns:
            ex_contexts (list[str]): list of document texts
            ex_questions (list[str]): list of questions
            ex_answers (list[dict]): list of answers (dict type)
        """
        # extracting questions, context and long answers
        ex_questions = list(map(lambda x: x["text"], examples["question"]))
        ex_contexts = list(
            map(lambda x: " ".join(x["tokens"]["token"]), examples["document"])
        )
        ex_answers = [
            self.single_answer_mapping_train(x, y["long_answer"])
            for x, y in zip(examples["document"], examples["annotations"])
        ]
        return ex_contexts, ex_questions, ex_answers

    def prepare_train_features(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[Dict[str, Any]],
    ) -> DatasetDict:
        """
        mapping function to prepare tokenized train examples
        Args:
            contexts (list):  list of document texts
            questions (list): list of questions
            answers (list): list of answers (dict type)

        Returns:
            tokenized_examples (DatasetDict): tokenized train examples
        """
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            sample_answers = answers[sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(sample_answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = sample_answers["answer_start"][0]
                end_char = start_char + len(sample_answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # if start char of long answer is found,  set start position to start char, otherwise to cls_index(0)
                if (
                    offsets[token_start_index][0]
                    <= start_char
                    <= offsets[token_end_index][1]
                ):
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                else:
                    tokenized_examples["start_positions"].append(cls_index)

                # if end char of long answer is found,  set end position to end char, otherwise to cls_index(0)
                token_start_index = 0
                if (
                    offsets[token_start_index][0]
                    <= end_char
                    <= offsets[token_end_index][1]
                ):
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                else:
                    tokenized_examples["end_positions"].append(cls_index)

        return tokenized_examples

    def prepare_validation_features(self, examples: dict) -> DatasetDict:
        """
        mapping function to prepare tokenized validation examples
        Args:
            examples (dict): dictionary containing list of examples from validation set

        Returns:
            tokenized_examples (dict): tokenized train examples

        """
        contexts, questions, answers = self.answer_mapping_train(examples)
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
