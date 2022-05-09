from aif360.datasets import AdultDataset, GermanDataset, CompasDataset,BankDataset,MEPSDataset19
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import aif360
from skfeature.function.similarity_based import fisher_score
import numpy as np
import heapq
import copy


def PFfisher(datasetname, dataset_orig_train, protected_feature):
    from sklearn import preprocessing

    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    if datasetname == 'adult':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'income-per-year'], dataset_orig_train[
            'income-per-year']
    if datasetname == 'compas':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'two_year_recid'], \
                           dataset_orig_train['two_year_recid']
    if datasetname == 'bank':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'y'], \
                           dataset_orig_train['y']
    if datasetname == 'german':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'credit'], \
                           dataset_orig_train['credit']
    if datasetname == 'meps':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'UTILIZATION'], \
                           dataset_orig_train['UTILIZATION']
    y_pro = X_train[protected_feature]
    X_train = X_train.drop(protected_feature,axis=1)
    column_train = [column for column in X_train]
    X_train = np.array(X_train)
    y_pro = np.array(y_pro)
    y_train = np.array(y_train)
    min_max_scaler = preprocessing.MinMaxScaler()
    score = fisher_score.fisher_score(X_train, y_pro)
    score = min_max_scaler.fit_transform(score.reshape(-1, 1))
    score1 = fisher_score.fisher_score(X_train, y_train)
    score1 = min_max_scaler.fit_transform(score1.reshape(-1, 1))
    score_final = []
    for i in range(len(score)):
        score_final.append(list(score1[i] - 0.1 * score[i])[0])

    choose_list = np.argsort(score_final)[::-1]
    choose_list1 = []
    for i in choose_list:
        choose_list1.append(column_train[i])
    return choose_list1


