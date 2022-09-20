import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from core.data_processing.se_dataset import SelfExplanations
from scripts.mtl.mtl_bert_train import get_train_test_IDs

file2 = "/home/bogdan/projects/PASTEL/data/results/results_paraphrase_se_aggregated_dataset_withprev_v2_v3.csv"
file = "/data/results/results_paraphrase_se_aggregated_dataset_v2_v3.csv"

df = pd.read_csv(file2, delimiter=',')

feature_columns = df.columns.tolist()[39:]

train_IDs, test_IDs = get_train_test_IDs([])
df_train = df[df['ID'].isin(train_IDs)]
df_test = df[df['ID'].isin(test_IDs)]
for task in SelfExplanations.MTL_TARGETS:
    # eliminating unlabeled datapoints for this task
    df_train_filtered = df_train[df_train[task] != 9]
    df_test_filtered = df_test[df_test[task] != 9]

    y_train = df_train_filtered[task]
    y_test = df_test_filtered[task]
    if 0 not in np.unique(np.concatenate([y_train, y_test])):
        y_test -= 1
        y_train -= 1
    x_train = df_train_filtered[feature_columns]
    x_test = df_test_filtered[feature_columns]

    model = XGBClassifier()
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    print(task, accuracy_score(y_predict, y_test))
