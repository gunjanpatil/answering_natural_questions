[![CodeFactor](https://www.codefactor.io/repository/github/gunjanpatil/answering_natural_questions/badge?s=d7ff811865d408f8f0322e6a2b217755d971a604)](https://www.codefactor.io/repository/github/gunjanpatil/answering_natural_questions)
# Answering Natural Questions
Cortx challenge: Deep learning model for answering [Google's natural questions](https://ai.google.com/research/NaturalQuestions)

This project looks at solving the problem of answering long answer type questions from [Google's natural questions dataset](https://github.com/google-research-datasets/natural-questions) using hugging face transformers.

When given a test example with document tokens, question text, and a list of long answer candidates, the model should predict which long answer candidate in the list most accurately answers the question.

The broad idea is to finetune pre-trained hugging face transformer-based question answering models to answer questions based on Wikipedia pages.

#### Version 1: Uniform sampling
- **Idea**: The model is trained using examples with a positive long-answer candidate and a uniformly sampled negative long-answer candidate from the positive examples. The idea is to make the model learn which candidates are correct predictions and which candidates are incorrect.
- **Model finetuned**: bert-base-uncased
- **Scores**:
  1. long-best-threshold-f1: 0.4608
  2. long-best-threshold-precision: 0.4157
  3. long-best-threshold-recall: 0.5169

#### Version 2: Better sampling
- **Idea**: The reason behind a low f1 score would be that the negative long-answer candidate sampled might not be the most challenging candidate against the positive candidate.
Thus, the next step would be to sample a hard negative candidate from a distribution that tells us the probability of hardness of each candidate. To get this distribution, we can use the model trained in version 0 that gives us the probability score of a positive candidate to mine hard negative examples.
- **Model to be finetuned**: deepset/bert-large-uncased-whole-word-masking-squad2
- **Scores**: pending

---
*System Requirements:*
1. Works with Python3
2. Trains and validates with a single GPU or a distributed system(multiple GPUs) - (Used lambda stack gpu cloud's 2x RTX 6000(24 GB) instance for this project)

*Major Packages Required*
1. [transformers~=4.1.1](https://github.com/huggingface/transformers, "huggingface transformers github")
2. [datasets~=1.1.3](https://github.com/huggingface/datasets, "huggingface datasets github")
3. [apex](https://github.com/NVIDIA/apex#quick-start, "nvidia apex")
4. torch~=1.7.1
---

#### Installation and Run:
1. **Clone** the repository 
    ```bash
    git clone https://github.com/gunjanpatil/answering_natural_questions.git
    cd answering_natural_questions
    ```
  
2. **Setup**: 
    
    To download, unzip and install all necessary packages,run setup.sh. This takes around 4-5 minutes, might vary depending on the network's downloading speed.
    ```. setup.sh```
    After this setup, you will be in the src directory of this repository. 
    
3. **Training**
    1. To train a model with train_v1.py, first modify training configurations in configs/args.json file according to your requirements.
    
        Mandatory modifications required:
          - *project_path*: path to your repository
        You can make changes to other arguments in the args file depending on your needs. You can also train using mixed precision by setting the fp16 argument to true.
      
    2. To launch training of a model using version 1, run the training script train_v1.py with a config file located in configs/args.json as follows:    
        ```bash
        cd src
        python3 -m torch.distributed.launch --nproc_per_node=<number_of_gpus_in_system> train_v1.py --configs=configs/args.json > train_v1_logs.txt
        ```
        Set nproc_per_node value to the number of GPUs in your system. This command also works with one GPU. The training configs used for this project are the same as the ones in configs/args.json. The model was trained on 2 RTX-6000 GPUs. The weights are saved in the output_dir mentioned in the configs file. 
    
4. **Validation**:
    1. To run validate using a mdel trained in version 1, run validate_v1.py to generate predictions on the validation dataset as follows:
        ```bash
        python3 validate_v1.py -d=../datasets/natural_questions_simplified/v1.0-simplified/nq-dev-all.jsonl -o=<path_to_directory_to_store_predictions_file -m=<model_name_or_path> -w=<path_to_saved_model_weights>
        ```
        You can also add a --fp16 argument further if your model was trained in mixed precision. On running the validation script, a predictions.json file will be generated in the output path given during execution of the script.
      
    2. Then, finally run google's evaluation script to generate f1 scores.
        ```bash
        python3 nq_eval.py --gold_path=../datasets/natural_questions_simplified/v1.0-simplified/nq-dev-all.jsonl.gz --predictions_path=<path_to_predictions.json_file > scores_predictions.txt
        ```
        All the scores will be written in the scores_predictions.txt file. *The scores for the bert-base-uncased model trained in version 1 is stored under predictions/bert-base-uncased/*
    
    
