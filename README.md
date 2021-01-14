[![CodeFactor](https://www.codefactor.io/repository/github/gunjanpatil/answering_natural_questions/badge?s=d7ff811865d408f8f0322e6a2b217755d971a604)](https://www.codefactor.io/repository/github/gunjanpatil/answering_natural_questions)
# Answering Natural Questions
Cortx challenge: Deep learning model for answering [Google's natural questions](https://ai.google.com/research/NaturalQuestions)

This project looks at solving the problem of answering long answer type questions from [Google's natural questions dataset](https://github.com/google-research-datasets/natural-questions) using hugging face transformers.

When given a test example with document tokens, question text, and a list of long answer candidates, the model should predict which long answer candidate in the list most accurately answers the question.

The broad idea is to finetune pre-trained hugging face transformer-based question answering models to answer questions based on Wikipedia pages.

#### Version 1: Uniform sampling
- Idea: The model is trained using examples with a positive long-answer candidate and a uniformly sampled negative long-answer candidate from the positive examples. The idea is to make the model learn which candidates are correct predictions and which candidates are incorrect.
- Model finetuned: bert-base-uncased
- long-best-threshold-f1: 0.46

#### Version 2: Better sampling
- Idea: The reason behind a low f1 score would be that the negative long-answer candidate sampled might not be the most challenging candidate against the positive candidate.
Thus, the next step would be to sample a hard negative candidate from a distribution that tells us the probability of hardness of each candidate. To get this distribution, we can use the model trained in version 0 that gives us the probability score of a positive candidate to mine hard negative examples.
- Model to be finetuned: deepset/bert-large-uncased-whole-word-masking-squad2
- long-best-threshold-f1: pending

## How to Run: 
#### TBA
