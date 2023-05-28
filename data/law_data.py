import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def law_data(seed=0):
    df = pd.read_csv("./data/law_data.csv", index_col=0)
    df["black"] = df["race"].map(lambda x: 1 if x == 'Black' else 0)
    df["white"] = df["race"].map(lambda x: 1 if x == 'White' else 0)
    df = df.drop(axis=1, columns=["race"])
    df = df[(df["black"]== 1) | (df["white"] == 1)]
    df["LSAT"] = df["LSAT"].astype(int)
    A = ["white","black"]
    df_train, df_test = train_test_split(df, random_state=seed, test_size=0.3)
    X_train = np.hstack((df_train[A], np.array(df_train["UGPA"]).reshape(-1, 1),np.array(df_train["LSAT"]).reshape(-1, 1)))
    y_train = df_train["ZFYA"]
    X_test = np.hstack((df_test[A], np.array(df_test["UGPA"]).reshape(-1, 1), np.array(df_test["LSAT"]).reshape(-1, 1)))
    y_test = df_test["ZFYA"]
    y_train_0 = y_train[(X_train[:, 1] == 1)]
    y_train_1 = y_train[(X_train[:, 0] == 1)]
    X_train_0 = X_train[X_train[:, 1] == 1,:]
    X_train_1 = X_train[X_train[:, 0] == 1,:]
    y_test_0 = y_test[(X_test[:, 1] == 1)]
    y_test_1 = y_test[(X_test[:, 0] == 1)]
    X_test_0 = X_test[X_test[:, 1] == 1, :]
    X_test_1 = X_test[X_test[:, 0] == 1, :]
    return X_train_0[:,2:],np.array(y_train_0), X_train_1[:,2:], np.array(y_train_1), X_test_0[:,2:], np.array(y_test_0), X_test_1[:,2:], np.array(y_test_1)
