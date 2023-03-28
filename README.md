# Paraphrasing ASsessment using TransfEr Learning (PASTEL)

The aim of this project is to present a set of models used for assessing paraphrase quality, as described in https://doi.org/10.3390/computers10120166.

Installation:
- from this project's root run "pip install -r requirements.txt"
- run "python -m spacy download en_core_web_lg"
- (only for training) create a "data" folder in the root of the project; copy data to the "data" folder within this project
  - for MSRP, copy msr_paraphrase_train.txt and msr_paraphrase_test.txt
  - for ULPC, copy the xls file containing the dataset
  - for the children dataset we can provide the file upon request


Replicating experiments:
- training:
  - preprocess the data by running core.paraphrase.input_process.py
  - run the baseline by running core.paraphrase.paraphrase_baseline.py
  - run paraphrase_(et|nn|bert) for training
- evaluating: 
  - for evaluating the bert model, you can use paraphrase_eval.py