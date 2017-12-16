"""
part2_model is a script used to evaludate different models and at last train and predict the label for the validation set
Input: train & validation processed data sets
Output: validation data set prediction, confusion table

This module should be executed after part2.py and from the same location as part2.py

Important Note: The process halts when a graph is showing. To compltete the process please close the graphs after
                you are done viewing them (thre are 2 graphs popping up during the process)

Important Note: Some of the operation are heavy and use all avilable processors. If the script is executed on a
                low  CPU machine, the process might fail.

Authors: Lavi.Lazarovitz (065957383) & Rasni Sharim (####)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
import pylab as P
from sklearn.utils import shuffle
from matplotlib import pyplot
from mpl_toolkits import mplot3d
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import Counter


def main():

    # ######################################## Data Loading ########################################
    # Loading the processed data from the local folder
    train_data = pd.read_csv("train_data_clean.csv", header=0)
    validation_data = pd.read_csv("validation_data_clean.csv", header=0)

    # Combining both data sets as they both will be used in k-fold cross validation training
    train_val_list = [train_data,validation_data]
    train_val_data = pd.concat(train_val_list)
    features = train_val_data.drop(['label'], axis=1).values
    target = train_val_data.label.values


    # ######################################## Model Evaluation ########################################
    # In this part we run the different models and assess (roughly) the model result
    # After this stage, we examine more thoroughly 2 chosen models


    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, features, target,cv=20, n_jobs=-1)
    print "########################   SVC    ############################"
    print "avg: %f" %(sum(scores) / 20.0)


    clf = LinearSVC(C=1.0, multi_class='ovr')
    scores = cross_val_score(clf, features, target,cv=20, n_jobs=-1)
    print "########################   LinearSVC    ############################"
    print "avg: %f" %(sum(scores) / 20.0)


    clf = OneVsOneClassifier(LinearSVC(C=1.0,random_state=0))
    scores = cross_val_score(clf, features, target,cv=20, n_jobs=-1)
    print "########################   OvO    ############################"
    print "avg: %f" %(sum(scores) / 20.0)


    clf =  GaussianNB()
    scores = cross_val_score(clf, features, target,cv=15, n_jobs=-1)
    print "########################   Naive Bayes    ############################"
    print "avg: %f" %(np.mean(scores))


    avg_score = []
    for splitter in range(2,20):
        clf = tree.DecisionTreeClassifier(min_samples_split=splitter)
        scores = cross_val_score(clf, features, target, cv=20, n_jobs=-1)
        average = sum(scores) / 20.0
        avg_score.append(average)
        # print "splitter=%d    average= %f" %(splitter,average)
    print "########################   Decision Trees    ############################"
    print "avg: %f" %(np.mean(avg_score))

    # Plotting to see what should be the minimum samples to split - this helps avoid over-fitting
    # Best results seems to be at around 5
    x = range(2,20)
    fig = pyplot.figure(figsize=(15,15))
    pyplot.plot(x,avg_score)
    pyplot.xlabel('Min samples to split')
    pyplot.ylabel('AVG score')
    pyplot.show()


    scrs = []
    for nbors in range (3,25):
        clf = KNeighborsClassifier(nbors)
        scores = cross_val_score(clf, features, target, cv=15, n_jobs=-1)
        avg = sum(scores) / 15.0
        scrs.append(avg)
        # print "neighbors=%d    average= %f" %(nbors,avg)
    print "########################   KNN    ############################"
    print "avg: %f" %(np.mean(scrs))

    # Plotting to see what should be the optimal k for the data (based results seems to be at k = 2)
    x = range(3,25)
    fig = pyplot.figure(figsize=(15,15))
    pyplot.plot(x,scrs)
    pyplot.xlabel('Number of neighbors')
    pyplot.ylabel('AVG score')
    pyplot.show()

    # ######################################## 2nd Model Evaluation ########################################
    # Best results were obtained by the Decision Trees and OneVSOne models. We examine those model closely using the
    # classification report

    # Generating classification report for Decision Trees
    clf = tree.DecisionTreeClassifier(min_samples_split=splitter)
    pred = cross_val_predict(clf, features, target, cv=30, n_jobs=-1)
    print "########################   Decision Tree    ############################"
    print(classification_report(target, pred, target_names=train_val_data.label.unique()))

    # Generating classification report for OvO
    clf = OneVsOneClassifier(LinearSVC(C=1.0,random_state=0))
    pred = cross_val_predict(clf, features, target, cv=30, n_jobs=-1)
    print "########################   OvO    ############################"
    print(classification_report(target, pred, target_names=train_val_data.label.unique()))

    # Looking at the data we noticed similar results for all parameters
    # Only difference we noticed was that the OvO has shown better results in predicting the larger parties
    # And this lead us to decide to take OvO with about 94.5% results


    # ######################################## Fit & Predict ########################################
    # We chose to use OvO to fit and predict as it has shown the best results for all scoring options
    # In this section, we will fit the data based on repeated_stratfied_kfold that takes into account the class sizes
    # and cross train and validate the data on different section of the data
    # The prediction will be of course only on the validation data - to see how the generalization works on the
    # original distribution

    # Creating 20 cross validation scenarios to train the data on. The data includes both train and validation
    k_fold = RepeatedStratifiedKFold(n_splits=2)

    # Training the model based on the created data sets
    print "Training model"
    for train_indices, test_indices in k_fold.split(features, target):
        clf = OneVsOneClassifier(LinearSVC(C=1.0, random_state=0)).fit(features[train_indices], target[train_indices])

    # preparing the validation set for prediction - removing the label from the data
    features = validation_data.drop(['label'], axis=1).values
    target = validation_data.label.values
    pred = clf.predict(features)
    print "Predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    print "Parties vote distribution"
    vote_count = Counter(pred)
    for party in vote_count:
        print party + ": " + str((vote_count[party] / 2000.0) * 100.0) + "%"

    # Saving prediction to csv
    df = pd.DataFrame(pred)
    df.to_csv("validation_prediction.csv",header=['label'] ,index=False)

    # Generating confusion matrix
    print(classification_report(target, pred, target_names=train_val_data.label.unique()))

    # printing number of errors
    errors = sum(1 for i, j in zip(target, pred) if i != j)
    percent_error = float(errors / 2000.0*100.0)
    print "Number of error %d; Error percent: %f" %(errors, percent_error)

if __name__ == '__main__':
    main()