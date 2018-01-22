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

# ######################################## Data Loading ########################################
# Loading the processed data from the local folder

feature_names = pd.read_csv("clean_features.csv", nrows=1).columns
predcited_labels = pd.read_csv("Vote_prediction.csv", header=0)
# print(feature_names)

data = pd.read_csv("clean_features.csv", header=0)
# validation_data = pd.read_csv("validation_data_clean.csv", header=0)

# validation_data = pd.read_csv("validation_data_clean.csv", header=0)
features = data.values
labels = predcited_labels.PredictVote.values

#     validation_data_no_lables = validation_data.drop(['label'], axis=1).values
#     validation_labels = validation_data.label.values

# k means, determine best k
distortions = []
silhouette_avg = []

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
    party = predcited_labels[['PredictVote']].values[idx][0]

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
    scores = cross_val_score(knnModel, features, labels, cv=5)
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

knnModel = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(features)
distances, indices = knnModel.kneighbors(features)
party_knn = {}

for idx in range(0,len(indices)):
    myIndices = indices[idx]
    myIdx = myIndices[0]
    myParty = predcited_labels[['PredictVote']].values[myIdx][0]
    if myParty not in party_knn:
         party_knn[myParty] = {}
    for otherIdx in myIndices[1:]:
        otherParty = predcited_labels[['PredictVote']].values[otherIdx][0]
        if otherParty not in party_knn[myParty]:
            party_knn[myParty][otherParty] = 0
        party_knn[myParty][otherParty] = party_knn[myParty][otherParty] + 1

print(party_knn)

