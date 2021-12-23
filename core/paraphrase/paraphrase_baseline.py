import pickle

import pandas as pd
from sklearn.metrics import classification_report, f1_score
import numpy as np

from core.utils.rcdoc import create_doc
from utils import get_words


DATA_FOLDER = "../../data/"
RESULTS_FOLDER = "../../data/results/"

doc1 = pd.read_excel(f'{DATA_FOLDER}Public_Paraphrase_Challenge(excel).xls')
doc2 = pd.read_csv(f'{RESULTS_FOLDER}results_paraphrase_train_data.csv', encoding='latin1', sep='\t')

print(len(doc1))
doc1 = doc1[doc1["Index"].isin(doc2["Index"])]
print(len(doc1))

metrics_list = [
    # "target_longer_T_R",
    "sen_len_diff_binary",
    "Stem_overlap",
    "TTR_binary",
    "LSA_bin",
    "F_ent_conf",
    "A_ent_conf",
    "R_ent_conf",
    "MED_binary"
]
targets_list = ["Paraphrase_Quality", "Paraphrase_quality_bin",
                "Semantic_Completeness", "Semantic_completeness_bin",
                "Entailment", "Entailment_bin",
                "Syntactic_Similarity", "Syntactic_similarity_bin",
                "Lexical_Similarity", "Lexical_similarity_bin",
                "Writing_Quality", "Writing_quality_bin"]

threshold_dict = {
    "Paraphrase_Quality": ("Paraphrase_quality_bin", True),
    "Semantic_Completeness": ("Semantic_completeness_bin", True),
    "Entailment": ("Entailment_bin", True),
    "Syntactic_Similarity": ("Syntactic_similarity_bin", True),
    "Lexical_Similarity": ("Lexical_similarity_bin", True),
    "Writing_Quality": ("Writing_quality_bin", True)
}

# targets_list = ["Paraphrase_quality_tri"]
doc_mod = doc1[["Index", "Clean_Target_Sentence", "Clean_Utterance",
                "Paraphrase_Quality", "Paraphrase_quality_bin",
                "Semantic_Completeness", "Semantic_completeness_bin",
                "Entailment", "Entailment_bin",
                "Syntactic_Similarity", "Syntactic_similarity_bin",
                "Lexical_Similarity", "Lexical_similarity_bin",
                "Writing_Quality", "Writing_quality_bin",
                "trn_test_val"
                ] + metrics_list]


test_df = doc_mod[doc_mod["trn_test_val"] == 3]

for t in targets_list:
    if t.endswith("_bin") or t.endswith("_tri"):
        print()
        print(f"{t}\tLow F1\tHigh F1")
        for m in metrics_list:
            metric = test_df[m]
            print(m, end="\t")
            if len(set(metric)) == 2:
                f1 = f1_score(test_df[t], metric, average=None)
                for f in f1:
                    print(round(f, 3), end='\t')
                print(round(sum(f1) / 2, 3), end='\t')
            elif max(metric) <= 1:
                f1 = f1_score(test_df[t], (metric > 0.5).astype(int), average=None)
                for f in f1:
                    print(round(f, 3), end='\t')
                print(round(sum(f1) / 2, 3), end='\t')
            else:
                print("pass", end="\t")
            print()

