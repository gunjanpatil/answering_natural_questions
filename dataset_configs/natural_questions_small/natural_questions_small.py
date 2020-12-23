# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Questions: A Benchmark for Question Answering Research."""

from __future__ import absolute_import, division, print_function

import json
import re

import apache_beam as beam
import six

import datasets


if six.PY2:
    import HTMLParser as html_parser  # pylint:disable=g-import-not-at-top

    html_unescape = html_parser.HTMLParser().unescape
else:
    import html  # pylint:disable=g-import-not-at-top

    html_unescape = html.unescape

_CITATION = """
@article{47761,
title	= {Natural Questions: a Benchmark for Question Answering Research},
author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
year	= {2019},
journal	= {Transactions of the Association of Computational Linguistics}
}
"""

_DESCRIPTION = """
The NQ corpus contains questions from real users, and it requires QA systems to
read and comprehend an entire Wikipedia article that may or may not contain the
answer to the question. The inclusion of real user questions, and the
requirement that solutions should read an entire page to find the answer, cause
NQ to be a more realistic and challenging task than prior QA datasets.
"""

_URL = "https://ai.google.com/research/NaturalQuestions/dataset"

_BASE_DOWNLOAD_URL = "https://storage.googleapis.com/natural_questions/v1.0"
_DOWNLOAD_URLS = {
    "train": ["%s/train/nq-train-%02d.jsonl.gz" % (_BASE_DOWNLOAD_URL, i) for i in range(5)],
    "validation": ["%s/dev/nq-dev-%02d.jsonl.gz" % (_BASE_DOWNLOAD_URL, i) for i in range(1)],
}


class NaturalQuestions(datasets.BeamBasedBuilder):
    """Natural Questions: A Benchmark for Question Answering Research."""

    VERSION = datasets.Version("0.0.4")
    SUPPORTED_VERSIONS = [datasets.Version("0.0.1"), datasets.Version("0.0.2")]
    print("Using custom version for small dataset")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document": {
                        "title": datasets.Value("string"),
                        "url": datasets.Value("string"),
                        "html": datasets.Value("string"),
                        "tokens": datasets.features.Sequence(
                            {"token": datasets.Value("string"), "is_html": datasets.Value("bool")}
                        ),
                    },
                    "question": {
                        "text": datasets.Value("string"),
                        "tokens": datasets.features.Sequence(datasets.Value("string")),
                    },
                    "annotations": datasets.features.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "long_answer": {
                                "start_token": datasets.Value("int64"),
                                "end_token": datasets.Value("int64"),
                                "start_byte": datasets.Value("int64"),
                                "end_byte": datasets.Value("int64"),
                            }
                        }
                    ),
                    "long_answer_candidates":datasets.features.Sequence(
                        {
                            "start_token": datasets.Value("int64"),
                            "end_token": datasets.Value("int64"),
                            "start_byte": datasets.Value("int64"),
                            "end_byte": datasets.Value("int64"),
                            "top_level": datasets.Value("bool"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, pipeline):
        """Returns SplitGenerators."""

        files = dl_manager.download(_DOWNLOAD_URLS)
        if not pipeline.is_local():
            print("is not local")
            files = dl_manager.ship_files_with_pipeline(files, pipeline)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepaths": files["validation"]},
            ),
        ]

    def _build_pcollection(self, pipeline, filepaths):
        """Build PCollection of examples."""

        def _parse_example(line):
            """Parse a single json line and emit an example dict."""
            ex_json = json.loads(line)
            html_bytes = ex_json["document_html"].encode("utf-8")

            def _parse_annotation(an_json):
                return {
                    # Convert to str since some IDs cannot be represented by datasets.Value('int64').
                    "id": str(an_json["annotation_id"]),
                    "long_answer": {
                        "start_token": an_json["long_answer"]["start_token"],
                        "end_token": an_json["long_answer"]["end_token"],
                        "start_byte": an_json["long_answer"]["start_byte"],
                        "end_byte": an_json["long_answer"]["end_byte"],
                    }
                }

            beam.metrics.Metrics.counter("nq", "examples").inc()
            # Convert to str since some IDs cannot be represented by datasets.Value('int64').
            id_ = str(ex_json["example_id"])
            return (
                id_,
                {
                    "id": id_,
                    "document": {
                        "title": ex_json["document_title"],
                        "url": ex_json["document_url"],
                        "html": ex_json["document_html"],
                        "tokens": [
                            {"token": t["token"], "is_html": t["html_token"]} for t in ex_json["document_tokens"]
                        ],
                    },
                    "question": {"text": ex_json["question_text"], "tokens": ex_json["question_tokens"]},
                    "annotations": [_parse_annotation(an_json) for an_json in ex_json["annotations"]],
                    "long_answer_candidates": [
                        {
                          "start_token": can['start_token'],
                          "end_token": can["end_token"],
                          "start_byte": can["start_byte"],
                          "end_byte": can["end_byte"],
                          "top_level": can["top_level"]
                        } for can in ex_json["long_answer_candidates"]
                    ]
                },
            )

        return (
            pipeline
            | beam.Create(filepaths)
            | beam.io.ReadAllFromText(compression_type=beam.io.textio.CompressionTypes.GZIP)
            | beam.Map(_parse_example)
        )