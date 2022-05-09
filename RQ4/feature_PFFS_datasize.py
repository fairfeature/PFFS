# Load all necessary packages
import sys
import numpy as np
import pandas
import pandas as pd
from sklearn.feature_selection import SelectKBest
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import CalibratedEqOddsPostprocessing
sys.path.append("../")
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset,BankDataset,MEPSDataset19
from aif360.metrics import BinaryLabelDatasetMetric, SampleDistortionMetric
from sklearn.feature_selection import chi2
from aif360.metrics import ClassificationMetric
from sklearn import tree
import statistics
import json
from sklearn.neighbors import KNeighborsClassifier
import os
from aif360.explainers import MetricTextExplainer, MetricJSONExplainer

import lib

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, accuracy_score
import random

## import dataset

from collections import OrderedDict
from aif360.metrics import ClassificationMetric

def collectdata(datasetname,protectedattribute):
    from sklearn.decomposition import PCA
    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from filter import PFfisher


    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(datasetname, protectedattribute)
    trainsizeratio = 0.1  # SET DATA SIZE
    originalfeatureset = dataset_orig.feature_names


    for turn in np.arange(0, 50, 1):

        seedr = turn
        print('================================================Turn:'+str(turn))

        dataset_orig_train_total, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed=seedr)
        dataset_orig_train_pred = dataset_orig_train_total.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split([trainsizeratio], shuffle=True, seed=seedr)

        featureset_index, featureset = PFfisher(datasetname, dataset_orig_train, protectedattribute)

        np.savetxt(datasetname + '_' + protectedattribute + '/feature_list/run_' + str(turn) + '.csv', featureset_index)


def runall():
    datasetnamelist = [['german', 'sex'], ['adult', 'sex'], ['adult', 'race'], ['bank', 'age'], ['german', 'age'], ['compas', 'sex'], ['compas', 'race'], ['meps', 'RACE']]
    for i in datasetnamelist:
        collectdata(i[0], i[1])


if __name__ == '__main__':
    runall()
