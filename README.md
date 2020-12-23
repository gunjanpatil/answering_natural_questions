[![CodeFactor](https://www.codefactor.io/repository/github/gunjanpatil/answering_natural_questions/badge?s=d7ff811865d408f8f0322e6a2b217755d971a604)](https://www.codefactor.io/repository/github/gunjanpatil/answering_natural_questions)
# Answering Natural Questions
cortx challenge answering natural questions

This project looks at how to finetune pretrained hugging face transfromers to anwer questions based on wiki pages. 

## things planned
#### verion 0: Simple pipeline
- simple dataset creator
- basic preprocessor
- basic trainer
- basic postprocessing
- eval script

#### version 1: Better postprocessing
add iou or other similarity metrics to get better mapping to long answer candidates

#### version 2: Better token parsing
filer and parse tokens making use of the structure inherent to html docs

#### version 3: New approach 
take each long answer and send to transfromer predict probablity of it haveing the answer to a given question choose one with highest prob


## How to Run: 
#### TBA
