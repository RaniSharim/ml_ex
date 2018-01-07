
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def main():
    # ######################################## Data Loading ########################################
    # Loading the processed data from the local folder
    train_data = pd.read_csv("train_data_clean.csv", header=0)
    # validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    features = train_data.drop(['label'], axis=1).values

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

    k = 11
    cluster_counts = {}
    party_counts = {}
    kmeanModel = KMeans(n_clusters=k).fit(features)
    cluster_labels = kmeanModel.fit_predict(features)
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
        
    
main()
