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

dataset = CompasDataset()

def filter(dataset, protected_feature):
    dataset_orig = dataset
    feature_list = dataset_orig.feature_names
    #print(feature_list)
    #print(len(feature_list))
    dataset_orig = dataset_orig.convert_to_dataframe()[0]
    scaler = MinMaxScaler()
    dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
    #print(dataset_orig)
    race = dataset_orig[protected_feature].values
    choose_list = []
    p_list = []
    for i in feature_list:
        if i != protected_feature:
            attri = dataset_orig[i].values
            print(attri)
            from scipy.stats import pearsonr
            p = pearsonr(race, attri)
            if p[1] < 0.05 and p[0] > 0:
                choose_list.append(i)
                p_list.append(p[0])
    print(p_list)
    #print(choose_list)
    #print(len(choose_list))
    return choose_list

#choose_list = filter(dataset,'sex')
#print(choose_list)

def laplacian(datasetname, dataset_orig_train, k, protected_feature):
    '''
    dataset_orig = dataset
    feature_list = dataset_orig.feature_names
    dataset_orig = dataset_orig.convert_to_dataframe()[0]
    male = dataset_orig[dataset_orig[protected_feature] == 1]
    print(male)
    '''
    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    print(dataset_orig_train)
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
    column_train = [column for column in X_train]
    print(X_train)
    X_train = np.array(X_train)
    print(X_train)
    y_train = np.array(y_train)
    print(y_train)
    score = fisher_score.fisher_score(X_train, y_train)
    print(score)
    idx = fisher_score.feature_ranking(score)
    print(idx)
    choose_list = list(idx[0:k])
    print(choose_list)
    for i in choose_list:
        if column_train[i] == protected_feature:
            print(i)
            choose_list = list(idx[0:k+1])
            choose_list.remove(i)
    #print(idx[0:5])
    print(choose_list)
    choose_list1 = []
    for i in choose_list:
        choose_list1.append(column_train[i])
    return choose_list1

def laplacian2(datasetname, dataset_orig_train, k, protected_feature):
    '''
    dataset_orig = dataset
    feature_list = dataset_orig.feature_names
    dataset_orig = dataset_orig.convert_to_dataframe()[0]
    male = dataset_orig[dataset_orig[protected_feature] == 1]
    print(male)
    '''
    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    print(dataset_orig_train)
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
    X_train.drop(protected_feature,axis=1)
    column_train = [column for column in X_train]
    print(X_train)
    X_train = np.array(X_train)
    print(X_train)
    y_pro = np.array(y_pro)
    print(y_pro)
    score = fisher_score.fisher_score(X_train, y_pro)
    print(score)
    idx = fisher_score.feature_ranking(score)
    print(idx)
    choose_list = list(idx[0:k])
    print(choose_list)
    for i in choose_list:
        if column_train[i] == protected_feature:
            print(i)
            choose_list = list(idx[0:k+1])
            choose_list.remove(i)
    #print(idx[0:5])
    print(choose_list)
    choose_list1 = []
    for i in choose_list:
        choose_list1.append(column_train[i])
    return choose_list1

def fisher3(datasetname, dataset_orig_train, protected_feature):
    from sklearn import preprocessing
    '''
    dataset_orig = dataset
    feature_list = dataset_orig.feature_names
    dataset_orig = dataset_orig.convert_to_dataframe()[0]
    male = dataset_orig[dataset_orig[protected_feature] == 1]
    print(male)
    '''
    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    #print(dataset_orig_train)
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
    #print(X_train)
    X_train = np.array(X_train)
    #print(X_train)
    y_pro = np.array(y_pro)
    #print(y_pro)
    y_train = np.array(y_train)
    min_max_scaler = preprocessing.MinMaxScaler()
    score = fisher_score.fisher_score(X_train, y_pro)
    score = min_max_scaler.fit_transform(score.reshape(-1, 1))
    score1 = fisher_score.fisher_score(X_train, y_train)
    score1 = min_max_scaler.fit_transform(score1.reshape(-1, 1))
    print(score)
    score_final = []
    for i in range(len(score)):
        score_final.append(list(score1[i] - 0.1 * score[i])[0]) #性能数据-公平数据
    print(score_final)

    choose_list = np.argsort(score_final)[::-1]
    choose_list1 = []
    for i in choose_list:
        choose_list1.append(column_train[i])
    return choose_list, choose_list1


def slopeTest(protected_feature, datasetname, dataset_orig_train, choose_list):
    from scipy import stats
    dataset_orig_train = dataset_orig_train.convert_to_dataframe()[0]
    #print(dataset_orig_train)
    if datasetname == 'adult':
        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'income-per-year'], \
                           dataset_orig_train[
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
    x_pro = X_train[protected_feature]

    slope_list = []
    intercept_list = []
    rvalue_list = []
    pvalue_list = []

    for i in choose_list:
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x_pro, X_train[i])
        rvalue_list.append(rvalue)
        pvalue_list.append(pvalue)
        slope_list.append(slope)
        intercept_list.append(intercept)


    return rvalue_list, pvalue_list, slope_list, intercept_list

