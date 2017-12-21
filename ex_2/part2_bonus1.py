"""
part2_bonus1 is a script used to automatically evaludate different models 
and choose the one with the lowest error level

Authors: Lavi.Lazarovitz (065957383) & Aharon Sharim (052328523)
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

class RunModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.score = 0

    def train(self, cv, features, target):
        scores = cross_val_score(self.model, features, target,cv=cv, n_jobs=-1)
        self.score = sum(scores) / cv
    
    def stam(self):
        print("a")

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


    svcModel = RunModel('svm', svm.SVC(kernel='linear', C=1))
    linearSvcModel = RunModel('Linear svc', LinearSVC(C=1.0, multi_class='ovr'))
    oneVsOneModel = RunModel('One Vs One', OneVsOneClassifier(LinearSVC(C=1.0,random_state=0)))
    guassianNBModel = RunModel('GaussianNB', GaussianNB())
    treeModel = RunModel('Decision Tree', tree.DecisionTreeClassifier(min_samples_split=5))
    nearestNighboarsModel = RunModel('KNeighborsClassifier', KNeighborsClassifier(3))

    classifiers = [svcModel, linearSvcModel, oneVsOneModel, guassianNBModel, treeModel, nearestNighboarsModel]

    # ######################################## Model Evaluation ########################################
    # In this part we run the different models and assess (roughly) the model result
    # After this stage, we examine more thoroughly 2 chosen models

    print("Training classifiers")
    for classifier in classifiers:
        print ("Training: " + classifier.name)
        classifier.train(20, features, target)

    # Sort the models by scores
    classifiersSorted = sorted(classifiers, key = lambda model: model.score, reverse=True)

    print("")
    print ("Results:")
    # Print in order
    for model in classifiersSorted:
        print(model.name + " " + str(model.score))

    # ######################################## 2nd Model Evaluation ########################################
    # Best results were obtained by the Decision Trees and OneVSOne models. We examine those model closely using the
    # classification report
    print("")
    print("Taking a closer look at top 2 classifiers")

    topModels = classifiersSorted[:2]
    for model in classifiers:
        print ("Re-training: " + classifier.name)
        model.train(30, features, target)

    # Sort the models by scores
    classifiersSorted = sorted(topModels, key = lambda model: model.score, reverse=True)
    
    # Print in order
    for model in classifiersSorted:
        print(model.name + " " + str(model.score))
    
    # Take most accurate model 
    selectedModel = classifiersSorted[0]

    print("")
    print ("Selected model:")
    print (selectedModel.name)

    # Creating 20 cross validation scenarios to train the data on. The data includes both train and validation
    k_fold = RepeatedStratifiedKFold(n_splits=2)

    # Training the model based on the created data sets
    print ("Training: " + selectedModel.name)
    print ("Training model")
    for train_indices, test_indices in k_fold.split(features, target):
        clf = selectedModel.model.fit(features[train_indices], target[train_indices])

    # preparing the validation set for prediction - removing the label from the data
    features = validation_data.drop(['label'], axis=1).values
    target = validation_data.label.values
    pred = clf.predict(features)
    print ("Predicted winner is %s" %(Counter(pred).most_common(1)[0][0]))

    print ("Parties vote distribution")
    vote_count = Counter(pred)
    for party in vote_count:
        print (party + ": " + str((vote_count[party] / 2000.0) * 100.0) + "%")

    # Saving prediction to csv
    df = pd.DataFrame(pred)
    df.to_csv("validation_prediction.csv",header=['label'] ,index=False)

    # Generating confusion matrix
    print(classification_report(target, pred, target_names=train_val_data.label.unique()))

    # printing number of errors
    errors = sum(1 for i, j in zip(target, pred) if i != j)
    percent_error = float(errors / 2000.0*100.0)
    print ("Number of error %d; Error percent: %f" %(errors, percent_error))

if __name__ == '__main__':
    main()
