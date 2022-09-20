import pandas as pd
import numpy
from core.data_processing.input_process import process_dataset
from core.data_processing.se_dataset import SelfExplanations


def clean_csv(file):
    df = pd.read_csv(file, delimiter='\t')

    # eliminate None entries
    df = df.replace('None', 0)

    # eliminating constant columns
    df = df.loc[:, (df != df.iloc[0]).any()]

    # normalize N/A datapoints
    for val in SelfExplanations.MTL_TARGETS:
        df[val][df[val] == 'BLANK '] = 9
        df[val][df[val] == 'BLANK'] = 9
        df[val][df[val] == 'blANK'] = 9
        df[val] = df[val].astype(int)

    #  fixing dtype errors
    feature_columns = df.columns.tolist()[38:]
    removable_cols = []
    for i, column in enumerate(feature_columns):
        if df[column].dtype.type not in [numpy.int64, numpy.float64]:
            try:
                df.loc[:,column] = df[column].astype(float)
                print("b", i, df[column].dtype.type)
            except:
                removable_cols.append(column)
                print("b", column, i, df[column].dtype.type, df[column].unique())
    print(f"Removed {removable_cols} because of datatype issues")
    df = df.drop(columns=removable_cols)

    feature_columns = df.columns.tolist()[39:]

    df.to_csv(f"{file[:-4]}_v2.csv")
    return df

filename = process_dataset("se_aggregated_dataset", use_prev_sentence=True)
clean_csv(filename)
filename = process_dataset("se_aggregated_dataset", use_prev_sentence=False)
clean_csv(filename)

