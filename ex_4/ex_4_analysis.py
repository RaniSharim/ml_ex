
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

def main():
    # ######################################## Data Loading ########################################
    # Loading the processed data from the local folder
    train_data = pd.read_csv("train_data_clean.csv", header=0)
    validation_data = pd.read_csv("validation_data_clean.csv", header=0)

    # validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    features = train_data.drop(['label'], axis=1).values
    train_labels = train_data.label.values
    validation_data_no_lables = validation_data.drop(['label'], axis=1).values
    validation_labels = validation_data.label.values
    # # k means determine k
    # distortions = []
    # silhouette_avg= []

    # K = range(2,20)
    # for k in K:
    #     print('Training '+str(k))
    #     kmeanModel = KMeans(n_clusters=k).fit(features)
    #     kmeanModel.fit(features)
    #     distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])
    #     cluster_labels = kmeanModel.fit_predict(features)
    #     silhouette_avg.append(silhouette_score(features, cluster_labels))

    # # Plot the elbow
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()

    #  # Plot the silhouette
    # plt.plot(K, silhouette_avg, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Silhouette')
    # plt.title('The Silhouette Method showing the optimal k')
    # plt.show()

    # k = 11
    # cluster_counts = {}
    # party_counts = {}
    # kmeanModel = KMeans(n_clusters=k).fit(features)
    # cluster_labels = kmeanModel.fit_predict(features)
    # for idx in range(0,len(cluster_labels)):
    #     cluster = str(cluster_labels[idx])
    #     party = train_data[['label']].values[idx][0]
        
    #     if cluster not in cluster_counts:
    #         cluster_counts[cluster] = {}

    #     if party not in cluster_counts[cluster]:
    #         cluster_counts[cluster][party] = 0
        
    #     cluster_counts[cluster][party] = cluster_counts[cluster][party] + 1

    #     if party not in party_counts:
    #         party_counts[party] = {}
        
    #     if cluster not in party_counts[party]:
    #         party_counts[party][cluster] = 0
        
    #     party_counts[party][cluster] = party_counts[party][cluster] + 1

    # print(cluster_counts)
    # print(party_counts)

    # accuracy = []
    # K = range(1,10)
    # for k in K:
    #     print('Training '+str(k))
    #     knnModel = KNeighborsClassifier(n_neighbors=k)
    #     scores = cross_val_score(knnModel, features, train_labels, cv=5)
    #     accuracy.append(sum(scores) / 5.0)
    #     # prediction = knnModel.predict(validation_data_no_lables)
    #     # accuracy.append(accuracy_score(validation_labels, prediction))

    # # Plot the accuricy
    # plt.plot(K, accuracy, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy for knn')
    # plt.show()

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

main()
