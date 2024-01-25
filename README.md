# SemEval-2024, Task 8: Multigenerator, Multidomain, and Multilingual Black-Box Machine-Generated Text Detection
## Documents

 1. State of the art discussion:
	 - https://docs.google.com/document/d/1LCqaNAUMEuTIV0ES5ddY38AVKUpwsCyBNQ0MmCf8H_4/edit#heading=h.10y221qa3z0e

## Possible approaches
1. Train from scratch a model:
   * NLP_from_scratch.ipynb
2. Fine-tuned existing trained mode:
   * NLP_FineTuning_BertCased.ipynb
   * NLP_FineTuning_Roberta.ipynb
3. Fine-tuned with unfreezing layers:
   * NLP_CustomModel-unfreeze.ipynb
4. Zero-shot:
   * Zero-shot.ipynb
5. Stats about training/testing data:
   * NLP_Stats.ipynb
##  Challenges
- big dataset to train on -> more time and resources needed
