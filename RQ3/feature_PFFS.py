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



def collectdata(datasetname,protectedattribute,datapath):
    from sklearn.decomposition import PCA
    from aif360.datasets import BinaryLabelDataset, StructuredDataset
    from filter import PFfisher

    writefile = open(datapath, 'w')
    writefile.write('datasetname' + ','
                    + 'turn' + ','
                    + 'trainsizeratio' + ','
                    + 'featurenum' + ','
                    + 'depth' + ','
                    + 'train.mean_difference' + ','
                    + 'testpred.accuracy' + ','
                    + 'testpred.recall' + ','
                    + 'testpred.precision' + ','
                    + 'testpred.f1' + ','
                    + 'testpred.false_alarm' + ','
                    + 'testpred.equal_opportunity_difference' + ','
                    + 'testpred.statistical_parity_difference' + ','
                    + 'testpred.average_abs_odds_difference' + ','
                    + 'testpred.disparate_impact' + ','
                    + '\n'
                    )

    dataset_orig, privileged_groups, unprivileged_groups = lib.get_data(datasetname, protectedattribute)
    trainsizeratio = 1.0
    label_name = dataset_orig.label_names[0]
    originalfeatureset = dataset_orig.feature_names

    for turn in np.arange(0, 50, 1):

        seedr = random.randint(0, 1000)
        print('================================================Turn:'+str(turn))

        dataset_orig_train_total, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed=seedr)
        dataset_orig_train_pred = dataset_orig_train_total.copy(deepcopy=True)
        dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
        dataset_orig_train, dataset_orig_mmm = dataset_orig_train_total.split([trainsizeratio], shuffle=True, seed=seedr)

        featureset = PFfisher(datasetname, dataset_orig_train, protectedattribute)
        featureset.append(protectedattribute)
        featuresubset_init = []

        protected_attribute_names = [protectedattribute]
        for each in protected_attribute_names:
            protected_attribute_index = featureset.index(each)
        for each in protected_attribute_names:
            featureset.remove(each)

        featurenumlist = np.arange(1, len(originalfeatureset), 1)
        depthlist = [10]

        for numfeatures in featurenumlist:

            featuresubset = list(np.copy(featuresubset_init))
            coveredfeaturelist = featuresubset[:]
            for feature in featureset:
                if len(coveredfeaturelist) == numfeatures:
                    break

                thisfeaturestring = feature
                coveredfeaturelist.append(thisfeaturestring)
                featuresubset.append(originalfeatureset.index(feature))

            scale_orig = StandardScaler()
            featuresubset = list(set(featuresubset))
            X_train_fullfeature = scale_orig.fit_transform(dataset_orig_train.features)
            y_train = dataset_orig_train.labels.ravel()
            X_train = X_train_fullfeature[:, featuresubset]

            metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)

            for depth in depthlist:

                lmod = tree.DecisionTreeClassifier(max_depth=depth)
                lmod.fit(X_train, y_train)

                fav_idx = np.where(lmod.classes_ == dataset_orig_train_total.favorable_label)[0][0]
                y_train_pred_prob = lmod.predict_proba(X_train)[:, fav_idx]

                X_test_fullfeature = scale_orig.transform(dataset_orig_test.features)
                X_test = X_test_fullfeature[:, featuresubset]
                y_test_pred_prob = lmod.predict_proba(X_test)[:, fav_idx]

                class_thresh = 0.5
                dataset_orig_train_pred.scores = y_train_pred_prob.reshape(-1, 1)
                dataset_orig_test_pred.scores = y_test_pred_prob.reshape(-1, 1)

                y_test_pred = np.zeros_like(dataset_orig_test_pred.labels)
                y_test_pred[y_test_pred_prob >= class_thresh] = dataset_orig_test_pred.favorable_label
                y_test_pred[~(y_test_pred_prob >= class_thresh)] = dataset_orig_test_pred.unfavorable_label
                dataset_orig_test_pred.labels = y_test_pred

                cm_transf_test = ClassificationMetric(dataset_orig_test, dataset_orig_test_pred,
                                                      unprivileged_groups=unprivileged_groups,
                                                      privileged_groups=privileged_groups)

                writefile.write(datasetname+','
                                +str(turn)+','
                                +str(trainsizeratio)+','
                                +str(numfeatures)+','
                                +str(depth)+','
                                +str(metric_orig_train.mean_difference())+','
                                +str(cm_transf_test.accuracy())+','#6
                                + str(cm_transf_test.recall()) + ','
                                + str(cm_transf_test.precision()) + ','
                                + str((2 * cm_transf_test.recall() * cm_transf_test.precision())/(cm_transf_test.precision() + cm_transf_test.recall())) + ','
                                + str(cm_transf_test.false_positive_rate()) + ','
                                +str(cm_transf_test.equal_opportunity_difference())+','
                                +str(cm_transf_test.statistical_parity_difference())+','
                                +str(cm_transf_test.average_abs_odds_difference())+','
                                +str(cm_transf_test.disparate_impact())+','
                                +'\n'
                                )
    writefile.close()

def drawFig(datasetname,protectedattribute,filepath):
    newpath = filepath.replace('.csv','_average.csv')

    readfile = open(newpath)
    lines = readfile.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip(',\n')
    metric_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    false_list = []
    equaloppfairmetric_list = []
    statislist = []
    averageoddlist = []
    disparate_impactlist = []
    traindifflist = []

    divia_add_metric_list = []
    divia_add_recall_list = []
    divia_add_precision_list = []
    divia_add_f1_list = []
    divia_add_false_list = []
    divia_add_equaloppfairmetric_list = []
    divia_add_statislist = []
    divia_add_averageoddlist = []
    divia_add_disparate_impactlist = []
    divia_add_traindifflist = []

    divia_sub_metric_list = []
    divia_sub_recall_list = []
    divia_sub_precision_list = []
    divia_sub_f1_list = []
    divia_sub_false_list = []
    divia_sub_equaloppfairmetric_list = []
    divia_sub_statislist = []
    divia_sub_averageoddlist = []
    divia_sub_disparate_impactlist = []
    divia_sub_traindifflist = []

    for thisline in lines:
        if 'trainsizeratio' in thisline:
            continue
        splits = thisline.split(',')
        feature = splits[1]
        if feature == '1' or feature == '2':
            continue
        traindifflist.append((float(splits[3])))
        metric_list.append((float(splits[4])))
        recall_list.append((float(splits[5])))
        precision_list.append((float(splits[6])))
        f1_list.append(float(splits[7]))
        false_list.append((float(splits[8])))
        equaloppfairmetric_list.append((float(splits[9])))
        statislist.append((float(splits[10])))
        averageoddlist.append((float(splits[11])))
        disparate_impactlist.append((float(splits[12])))

        divia_add_traindifflist.append((float(splits[13])+float(splits[3])))
        divia_add_metric_list.append((float(splits[14])+float(splits[4])))
        divia_add_recall_list.append((float(splits[15])+float(splits[5])))
        divia_add_precision_list.append((float(splits[16])+float(splits[6])))
        divia_add_f1_list.append((float(splits[17])+float(splits[7])))
        divia_add_false_list.append((float(splits[18])+float(splits[8])))
        divia_add_equaloppfairmetric_list.append((float(splits[19])+float(splits[9])))
        divia_add_statislist.append((float(splits[20])+float(splits[10])))
        divia_add_averageoddlist.append((float(splits[21])+float(splits[11])))
        divia_add_disparate_impactlist.append((float(splits[22])+float(splits[12])))

        divia_sub_traindifflist.append(float(splits[3]) - float(splits[13]))
        divia_sub_metric_list.append(float(splits[4]) - float(splits[14]))
        divia_sub_recall_list.append((float(splits[5]) - float(splits[15])))
        divia_sub_precision_list.append((float(splits[6]) - float(splits[16])))
        divia_sub_f1_list.append((float(splits[7]) - float(splits[17])))
        divia_sub_false_list.append((float(splits[8]) - float(splits[18])))
        divia_sub_equaloppfairmetric_list.append(float(splits[9]) - float(splits[19]))
        divia_sub_statislist.append(float(splits[10]) - float(splits[20]))
        divia_sub_averageoddlist.append(float(splits[11]) - float(splits[21]))
        divia_sub_disparate_impactlist.append(float(splits[12]) - float(splits[22]))

    range_ = np.arange(3,len(statislist)+3,1)
    plt.switch_backend('agg')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    lines = []
    lines += ax1.plot(range_, statislist, '.-.',color='b', label='statistical parity', linewidth=2)
    #ax1.fill_between(range_, divia_add_statislist, divia_sub_statislist, facecolor='b', alpha=0.1)

    lines += ax1.plot(range_, averageoddlist, '--', color='black', label='average abs odds', linewidth=2)
    #ax1.fill_between(range_, divia_add_averageoddlist, divia_sub_averageoddlist, facecolor='orange', alpha=0.1)

    lines += ax1.plot(range_, equaloppfairmetric_list, '-', marker='o',color='r', label='equal opportunity', linewidth=2)
    #ax1.fill_between(range_, divia_add_equaloppfairmetric_list, divia_sub_equaloppfairmetric_list, facecolor='r',
                     #alpha=0.1)

    lines += ax1.plot(range_, disparate_impactlist, ':', color='green', label='disparate impact', linewidth=2)
    #ax1.fill_between(range_, divia_add_disparate_impactlist, divia_sub_disparate_impactlist, facecolor='green',
                     #alpha=0.1)

    lines += ax1.plot(range_, metric_list, '-.', label='accuracy', linewidth=2)
    #ax1.fill_between(range_, divia_add_metric_list, divia_sub_metric_list, facecolor='y', alpha=0.1)

    lines += ax1.plot(range_, recall_list,  label='recall', linewidth=2)
    #ax1.fill_between(range_, divia_add_recall_list, divia_sub_recall_list, facecolor='y', alpha=0.1)

    lines += ax1.plot(range_, precision_list,  label='precision', linewidth=2)
    #ax1.fill_between(range_, divia_add_precision_list, divia_sub_precision_list, alpha=0.1)

    lines += ax1.plot(range_, f1_list,   label='f1-score', linewidth=2)
    #ax1.fill_between(range_, divia_add_f1_list, divia_sub_f1_list, alpha=0.1)

    lines += ax1.plot(range_, false_list,   label='false_alarm', linewidth=2)
    #ax1.fill_between(range_, divia_add_false_list, divia_sub_false_list, alpha=0.1)

    ax1.set_title(datasetname+' - '+protectedattribute.lower() +' - OM', fontsize=25, fontweight='bold')
    ax1.set_xlabel('', fontsize=28, fontweight='bold')
    ax1.set_ylabel('', color='black', fontsize=28, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=28)
    ax1.yaxis.set_tick_params(labelsize=28)

    ax1.set_ylim((-0.1, 1.0))
    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if 'adult' or 'compas' in datasetname:
        plt.xticks(np.arange(min(range_), max(range_) + 2, int(len(range_)/3)))

    if 'bank' or 'german' in datasetname:
        plt.xticks(np.arange(min(range_), max(range_) + 2, int(len(range_)/3)))
    if 'meps' in datasetname:
        plt.xticks(np.arange(min(range_), max(range_) + 2, int(len(range_)/3)))

    plt.savefig('./plots/'+datasetname+'-'+protectedattribute+'-fn.pdf',bbox_inches='tight')

    plt.legend()

    plt.show()

def runall():
    datasetnamelist = [['german', 'age'], ['adult','race'], ['adult','sex'], ['compas', 'race'], ['compas', 'sex'], ['bank', 'age'], ['german', 'sex'], ['meps', 'RACE']]
    for i in datasetnamelist:
        filepath = './results/' + i[0] + '-' + i[1] + '-PFFS.csv'
        collectdata(i[0], i[1], filepath)
        lib.get_average(filepath)
        drawFig(i[0], i[1], filepath)

if __name__ == '__main__':
    runall()
