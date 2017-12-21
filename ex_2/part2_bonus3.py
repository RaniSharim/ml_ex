
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
    k_fold = RepeatedStratifiedKFold(n_splits=2)

    # Combining both data sets as they both will be used in k-fold cross validation training
    train_val_list = [train_data,validation_data]
    train_val_data = pd.concat(train_val_list)
    features = train_val_data.drop(['label'], axis=1).values
    target = train_val_data.label.values

    print "Training model"
    for train_indices, test_indices in k_fold.split(features, target):
        clf = OneVsOneClassifier(LinearSVC(C=1.0, random_state=0)).fit(features[train_indices], target[train_indices])

    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Original predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Overall_happiness_score'] = validation_data['Overall_happiness_score'].map(lambda x: x+1.75)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Overall_happiness_score ++ predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Overall_happiness_score'] = validation_data['Overall_happiness_score'].map(lambda x: x-0.25)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Overall_happiness_score -- predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Will_vote_only_large_party_int'] = validation_data['Will_vote_only_large_party_int'].map(lambda x: 1)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Will_vote_only_large_party_int = 1 predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Will_vote_only_large_party_int'] = validation_data['Will_vote_only_large_party_int'].map(lambda x: 0)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Will_vote_only_large_party_int = 0 predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Garden_sqr_meter_per_person_in_residancy_area'] = validation_data['Garden_sqr_meter_per_person_in_residancy_area'].map(lambda x: x+2)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Garden_sqr_meter_per_person_in_residancy_area ++ predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Number_of_valued_Kneset_members'] = validation_data['Number_of_valued_Kneset_members'].map(lambda x: x-0.25)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Number_of_valued_Kneset_members -- predicted winner is %s" %(Counter(pred).most_common(1)[0][0])


if __name__ == '__main__':
    main()