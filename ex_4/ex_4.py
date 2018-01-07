
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
def main():
    # ######################################## Data Loading ########################################
    # Loading the processed data from the local folder
    train_data = pd.read_csv("train_data_clean.csv", header=0)
    # validation_data = pd.read_csv("validation_data_clean.csv", header=0)
    features = train_data.drop(['label'], axis=1).values

    # k means determine k
    distortions = []
    K = range(1,10)
    for k in K:
        print('Training '+str(k))
        kmeanModel = KMeans(n_clusters=k).fit(features)
        kmeanModel.fit(features)
        distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


main()
