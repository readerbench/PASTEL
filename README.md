# Paraphrasing ASsessment using TransfEr Learning (PASTEL)

The aim of this project is to present a set of models used for assessing paraphrase quality, as described in https://doi.org/10.3390/computers10120166.

Installation:
- install readerbench locally
  - clone the readerbench repository locally
  - within the root of the readerbench project run "pip install -r requirements.txt"
  - within the root of the readerbench project run "pip install -e ."
- copy data to the "data" folder within this project
  - for MSRP, copy msr_paraphrase_train.txt and msr_paraphrase_test.txt
  - for ULPC, copy the xls file containing the dataset
  - for the children dataset we can provide the file upon request
- from this project's root run "pip install -r requirements.txt"
- run "python -m spacy download en_core_web_lg"

Replicating experiments:
- preprocess the data by running core.paraphrase.input_process.py
- run the baseline by running core.paraphrase.input_process.py
- run paraphrase_(et|nn|bert) for training and testing the 3 models (ET, SN, BERT)
- Note: The current codebase should allow the user to recreate all the experiments presented in the paper. 
However, some tinkering with the scripts is necessary. This will be addressed in the next version.