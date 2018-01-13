"""
This is the model script. 
Important: The data_cleaning.py script needs to run first!!! 

Input: 3 csvs prepared for training and predicting
Output: Varius graphs and data tables needed to anaylze the elections

Authors: Lavi.Lazarovitz (065957383) & Aharon Sharim (052328523)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model_selection
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score
import matplotlib.cm as cm
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from collections import Counter

import math

def main():
    # ######################################## Data Loading ########################################
    # Loading the processed data from the local folder

    feature_names = pd.read_csv("validation_data_clean.csv", nrows=1).drop(['label'], axis=1).columns
    # print(feature_names)

    train_data = pd.read_csv("train_data_clean.csv", header=0)
    validation_data = pd.read_csv("validation_data_clean.csv", header=0)

    # validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    features = train_data.drop(['label'], axis=1).values
    train_labels = train_data.label.values

    validation_data_no_lables = validation_data.drop(['label'], axis=1).values
    validation_labels = validation_data.label.values

    # k means, determine best k
    distortions = []
    silhouette_avg= []

    K = range(2,20)
    for k in K:
        print('Training '+str(k))
        kmeanModel = KMeans(n_clusters=k).fit(features)
        kmeanModel.fit(features)
        distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])
        cluster_labels = kmeanModel.fit_predict(features)
        silhouette_avg.append(silhouette_score(features, cluster_labels))

    # Plot the elbow
    # This displays a graph, need to close it manually
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

     # Plot the silhouette
    # This displays a graph, need to close it manually
    plt.plot(K, silhouette_avg, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette')
    plt.title('The Silhouette Method showing the optimal k')
    plt.show()
  
  
    k = 11
    cluster_counts = {}
    party_counts = {}
    kmeanModel = KMeans(n_clusters=k).fit(features)
    cluster_labels = kmeanModel.fit_predict(features)
    
    print(kmeanModel.inertia_)
    maxDistance = 0

    distances = [[0 for x in range(11)] for y in range(11)] 

    for idx in range(0, 11):
        me = kmeanModel.cluster_centers_[idx]
        for idx2 in range(0, 11):
            other = kmeanModel.cluster_centers_[idx2]
            distance = 0
            for fIdx in range(1, len(me)):
                distance = distance + math.pow((me[fIdx] - other[fIdx]),2)
            distance = math.sqrt(distance)
            distances[idx][idx2] = distance
            if distance > maxDistance:
                maxDistance = distance
            #print str(distance)+",",
        #print ""

    for idx in range(0, 11):
        for idx2 in range(0, 11):
            distances[idx][idx2] = (distances[idx][idx2] - (maxDistance / 2))/(maxDistance / 2)

    for idx in range(0, 11):
        for idx2 in range(0, 11):
            if distances[idx][idx2] > 0:
                print "Far,",
            else:
                print "Close,",
        print ""

    for idx in range(0,len(cluster_labels)):
        cluster = str(cluster_labels[idx])
        party = train_data[['label']].values[idx][0]
        
        if cluster not in cluster_counts:
            cluster_counts[cluster] = {}

        if party not in cluster_counts[cluster]:
            cluster_counts[cluster][party] = 0
        
        cluster_counts[cluster][party] = cluster_counts[cluster][party] + 1

        if party not in party_counts:
            party_counts[party] = {}
        
        if cluster not in party_counts[party]:
            party_counts[party][cluster] = 0
        
        party_counts[party][cluster] = party_counts[party][cluster] + 1

    print(cluster_counts)
    print(party_counts)

    accuracy = []
    K = range(1,10)
    for k in K:
        print('Training '+str(k))
        knnModel = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knnModel, features, train_labels, cv=5)
        accuracy.append(sum(scores) / 5.0)
        # prediction = knnModel.predict(validation_data_no_lables)
        # accuracy.append(accuracy_score(validation_labels, prediction))

    # Plot the accuricy
    # This displays a graph, need to close it manually
    plt.plot(K, accuracy, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for knn')
    plt.show()

    knnModel = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(features)
    distances, indices = knnModel.kneighbors(features)
    party_knn = {}

    for idx in range(0,len(indices)):
        myIndices = indices[idx]
        myIdx = myIndices[0]
        myParty = train_data[['label']].values[myIdx][0]
        if myParty not in party_knn:
             party_knn[myParty] = {}
        for otherIdx in myIndices[1:]:
            otherParty = train_data[['label']].values[otherIdx][0]
            if otherParty not in party_knn[myParty]:
                party_knn[myParty][otherParty] = 0
            party_knn[myParty][otherParty] = party_knn[myParty][otherParty] + 1

    print(party_knn)

    gaussianNB =  GaussianNB()
    scores = cross_val_score(gaussianNB, features, train_labels,cv=5)
    print ("########################   Naive Bayes    ############################")
    print ("avg: %f" %(np.mean(scores)))


    gaussianFit =  gaussianNB.fit(features, train_labels)
   
    for idx in range(0, len(gaussianFit.classes_)):
                print(gaussianFit.classes_[idx])

    for featureIdx in range(0, len(gaussianNB.theta_[0])):
            print feature_names[featureIdx]+",",
            for idx in range(0, len(gaussianFit.classes_)):
                print str(round(gaussianNB.theta_[idx][featureIdx],3))+",",
                # print("Var: "+str(gaussianNB.sigma_[idx][featureIdx]))
            print ""

    # preparing the validation set for prediction - removing the label from the data
    k_fold = RepeatedStratifiedKFold(n_splits=2)

    # Training the model based on the created data sets
    print "Training model"
    for train_indices, test_indices in k_fold.split(features, train_labels):
        clf = OneVsOneClassifier(LinearSVC(C=1.0, random_state=0)).fit(features[train_indices],train_labels[train_indices])

    pred = clf.predict(validation_data_no_lables)
    count = Counter(pred)
    coalition = float(count['Purples']+count['Browns']+count['Whites']+count['Pinks'])
    size = float(len(validation_data))

    print "Original predicted winner is %s" %(count.most_common(1)[0][0])
    print "Original predicted coalition size is "+str(coalition/size*100) +"%"

    #Doing predictions after manipulations

    #Manipulations by features gained via tree
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

    # Features gained while looking at the bayesian model:
    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Overall_happiness_score'] = validation_data['Overall_happiness_score'].map(lambda x: x+1)
    validation_data['Garden_sqr_meter_per_person_in_residancy_area'] = validation_data['Garden_sqr_meter_per_person_in_residancy_area'].map(lambda x: x-1)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Overall_happiness_score ++ & Garden_sqr_meter_per_person_in_residancy_area -- predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

    validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    validation_data['Number_of_valued_Kneset_members'] = validation_data['Number_of_valued_Kneset_members'].map(lambda x: x-0.15)
    validation_data['Weighted_education_rank'] = validation_data['Weighted_education_rank'].map(lambda x: x+1)
    features_values = validation_data.drop(['label'], axis=1).values
    target_values = validation_data.label.values
    pred = clf.predict(features_values)
    print "Manipulated Number_of_valued_Kneset_members -- & Weighted_education_rank ++ predicted winner is %s" %(Counter(pred).most_common(1)[0][0])

main()
