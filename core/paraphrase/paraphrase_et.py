import pickle
import warnings
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import sklearn
import numpy as np

from core.paraphrase.utils import seed_everything

warnings.filterwarnings("ignore")

DATA_FOLDER = "../../data/"
RESULTS_FOLDER = "../../data/results/"

def classify_once(X_train, y_train, X_test, y_test, metrics = {}, RANDOM_SEED=13, title="", good_index = [], version = "ULPC"):
    clf_e = ExtraTreesClassifier(random_state=RANDOM_SEED, n_estimators=100)
    clf_e.fit(X_train, y_train)

    # y_pred_e = clf_e.predict(X_test)
    # print(classification_report(y_test, y_pred_e))

    # select top 100 features
    aux = [x for x in clf_e.feature_importances_]
    aux.sort(reverse=True)
    top100 = aux[99]
    top_indexes = [i for i in range(len(aux)) if clf_e.feature_importances_[i] >= top100]

    good_index = [good_index[i] for i in range(len(aux)) if clf_e.feature_importances_[i] >= top100]

    print(top_indexes)
    X_train = X_train[:, top_indexes]
    X_test = X_test[:, top_indexes]
    print("===")
    clf_e.fit(X_train, y_train)
    y_pred_e = clf_e.predict(X_test)

    print(classification_report(y_test, y_pred_e, digits=3))

    pickle.dump(clf_e, open(f"{RESULTS_FOLDER}et_{title}_model{version}.bin", "wb"))
    pickle.dump(good_index, open(f"{RESULTS_FOLDER}et_{title}_model_index{version}.bin", "wb"))
    print("===")
    metrics['accuracy_score']['extra_trees'].append(sklearn.metrics.accuracy_score(y_test, y_pred_e))
    metrics['precision_score']['extra_trees'].append(
        sklearn.metrics.precision_score(y_test, y_pred_e, average='weighted'))
    metrics['recall_score']['extra_trees'].append(
        sklearn.metrics.recall_score(y_test, y_pred_e, average='weighted'))
    metrics['confusion_matrix']['extra_trees'].append(
        sklearn.metrics.confusion_matrix(y_test, y_pred_e, normalize='true'))

def filter_normality(x):
    return np.quantile(x, 0.9) == min(x)

def filter_features(X_train, X_test, y_r_train):
    bad_index = []
    for col_id1 in range(len(X_train[0]) - 1):
        c1 = X_train[:, col_id1]
        if filter_normality(c1):
            bad_index.append(col_id1)
            continue
        if col_id1 in bad_index:
            continue
        multi_col_list = []
        for col_id2 in range(col_id1 + 1, len(X_train[0])):
            if col_id2 not in bad_index:
                c2 = X_train[:, col_id2]
                if np.corrcoef(c1.tolist(), c2.tolist())[0][1] > 0.9:
                    multi_col_list.append(col_id2)
        if len(multi_col_list) > 0:
            multi_col_list.append(col_id1)
            score_col_score = [np.corrcoef(X_train[:, id], y_r_train.reshape(-1))[0][1] for id in multi_col_list]
            extra = [multi_col_list[id] for id in range(len(score_col_score)) if id != np.argmax(score_col_score)]

            bad_index += extra
            # print(">", score_col_score)
            # print(">", multi_col_list)
            # print(">", extra)

    good_index = [i for i in range(len(X_train[0])) if i not in bad_index]
    print("Final split", len(bad_index), len(good_index), len(X_train[0]))
    X_train = X_train[:, good_index]
    X_test = X_test[:, good_index]

    return X_train, X_test, good_index

def train(version="ULPC"):
    USE_PAPER_SPLIT=True
    input = pd.read_csv(f'{DATA_FOLDER}results/results_paraphrase_train_data.csv', encoding='latin1', sep='\t')
    input["AvgWordWdPolysemy_source"] = 1
    input["AvgWordWdPolysemy_prod"] = 1
    cols = input.columns.tolist()[17:]
    source_cols = [col for col in cols if col.endswith("_source")]
    for col in source_cols:
        input[f"{col[:-7]}_diff"] = input[f"{col[:-7]}_source"] - input[f"{col[:-7]}_prod"]

    X = input.values[:, 17:]
    seed_everything(1234)

    metric_list = [
        ["Semantic_completeness", "Semantic_completeness_bin"],
        ["Paraphrase_quality", "Paraphrase_quality_tri"],
        ["Paraphrase_quality", "Paraphrase_quality_bin"],
        ["Syntactic_similarity", "Syntactic_similarity_bin"],
        ["Lexical_similarity", "Lexical_similarity_bin"]
    ]

    for metric_r, metric_c in metric_list:
        print("==========================================")
        print("============" + metric_c)
        print("==========================================")
        y_r = input[metric_r].values.reshape(-1, 1)
        y_c = input[metric_c].values.reshape(-1, 1)

        if version == "ULPC":
            X_train, X_test = X[input['trn_test_val'] == 1].astype(float), X[input['trn_test_val'] == 3].astype(float)
            y_r_train, y_r_test = y_r[input['trn_test_val'] == 1].astype(float), y_r[input['trn_test_val'] == 3].astype(float)
            y_c_train, y_c_test = y_c[input['trn_test_val'] == 1].astype(float), y_c[input['trn_test_val'] == 3].astype(float)
        else:
            X_train, X_test = X.astype(float), X[input['trn_test_val'] == 3].astype(float)
            y_r_train, y_r_test = y_r.astype(float), y_r[input['trn_test_val'] == 3].astype(float)
            y_c_train, y_c_test = y_c.astype(float), y_c[input['trn_test_val'] == 3].astype(float)

        X_train, X_test, good_index = filter_features(X_train, X_test, y_r_train)

        metrics_c = {
            "accuracy_score": {
                "extra_trees": [],
                "svc": [],
                "mlp": []
            },
            "precision_score": {
                "extra_trees": [],
                "svc": [],
                "mlp": []
            },
            "recall_score": {
                "extra_trees": [],
                "svc": [],
                "mlp": []
            },
            "confusion_matrix": {
                "extra_trees": [],
                "svc": [],
                "mlp": []
            }
        }
        classify_once(X_train, y_c_train, X_test, y_c_test, metrics_c, title=metric_c, good_index=good_index, version = version)
        print(metrics_c)


def validate(dataset, version="ULPC"):
    if dataset == "children" or dataset == "adults" or dataset in ["ordered", "random"]:
        df = pd.read_csv(f'{DATA_FOLDER}results_paraphrase_{dataset}_v3.csv', encoding='latin1', sep='\t')
        df = df[df["Paraphrase_quality_tri"] != 9]
        df = df[df["Semantic_completeness_bin"] != 9]
        cols = df.columns.tolist()[6:]

        source_cols = [col for col in cols if col.endswith("_source")]
        print(cols)
        print(source_cols)
        for col in source_cols:
            df[f"{col[:-7]}_diff"] = df[f"{col[:-7]}_source"] - df[f"{col[:-7]}_prod"]
        cols = df.columns.tolist()[6:]
        print(len(cols), cols)
        X = df.values[:, 6:]

        for feature in ["Paraphrase_quality_tri", "Paraphrase_quality_bin"]:#df.columns.tolist()[2:6]:
            model = pickle.load(open(f"{RESULTS_FOLDER}et_{feature}_model{version}.bin", "rb"))
            index = pickle.load(open(f"{RESULTS_FOLDER}et_{feature}_model_index{version}.bin", "rb"))
            print(index)
            filtered_feats = X[:, index]
            print("feats", "".join([f"{cols[i]}\t{model.feature_importances_[k]}\n" for k, i in enumerate(index)]))

            y_pred = model.predict(filtered_feats)
            print(y_pred)
            # y_test = df[feature].values.reshape(-1, 1)
            # print(classification_report(y_test, y_pred, digits=3))
            # # print(confusion_matrix(y_test, y_pred))
    else:
        df = pd.read_csv(f'{DATA_FOLDER}results_paraphrase_train_data_v2.csv', encoding='latin1', sep='\t')
        df = df[df['trn_test_val'] == 3]
        df_aux = pd.read_csv(f'{DATA_FOLDER}results_paraphrase_children_v2.csv', encoding='latin1', sep='\t')
        cols = df.columns.tolist()[17:]

        source_cols = [col for col in cols if col.endswith("_source")]
        print(cols)
        print(source_cols)
        for col in source_cols:
            df[f"{col[:-7]}_diff"] = df[f"{col[:-7]}_source"] - df[f"{col[:-7]}_prod"]
        cols = df.columns.tolist()[17:]

        print(len(cols), cols)
        X = df.values[:, 17:]

        for feature in ["Paraphrase_quality_bin"] + df_aux.columns.tolist()[2:6]:
            model = pickle.load(open(f"{RESULTS_FOLDER}et_{feature}_model{version}.bin", "rb"))
            index = pickle.load(open(f"{RESULTS_FOLDER}et_{feature}_model_index{version}.bin", "rb"))
            print(index)
            filtered_feats = X[:, index]
            y_pred = model.predict(filtered_feats)
            y_test = df[feature].values.reshape(-1, 1)
            print(feature)
            print(classification_report(y_test, y_pred, digits=3))
            print("=" * 20)


if __name__ == '__main__':
    train()