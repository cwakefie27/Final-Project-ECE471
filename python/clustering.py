import numpy as np
import sys
import matplotlib.pyplot as plt

class clusteringClassifier:
    def __init__(self, k_value=2, minkowski_p=2, max_iter=300):
        self.k_value = k_value
        self.minkowski_p = minkowski_p
        self.max_iter = max_iter

    def _minkowski_distance(self,u,v):
        if self.minkowski_p == np.inf:
            return float(max(np.abs(u-v)))
        else:
            return sum(abs(u-v)**(float(self.minkowski_p))) ** (1/float(self.minkowski_p))

    def _assign_to_centroids(self,data):
        centroids_changed = False
        for i in range(0,len(data)):
            min_distance = np.inf
            centroid_assignment_index = -1

            for j in range(0,len(self.centroids_)):
                distance = self._minkowski_distance(data[i],self.centroids_[j])
                if (distance < min_distance):
                    centroid_assignment_index = j
                    min_distance = distance

            if (self.data_centroid_indexes_[i] != centroid_assignment_index):
                centroids_changed = True
                self.data_centroid_indexes_[i] = centroid_assignment_index;

        return centroids_changed


    def _adjust_centroids(self,data):
        self.centroids_[:] = 0
        count = np.zeros(len(self.centroids_))
        for data_index,centroid_index in enumerate(self.data_centroid_indexes_):
            self.centroids_[centroid_index] += data[data_index]
            count[centroid_index] += 1
        #Divide by the count to adjust centroids
        self.centroids_ = self.centroids_/count.reshape((len(count),1))

    def fit(self,data):
        #Assign centroid data indexes
        self.centroids_ = np.array([np.array(data[i]) for i in np.random.randint(low=0, high=len(data), size=self.k_value)])
        self.data_centroid_indexes_ = [0] * len(data)

        while self._assign_to_centroids(data) == True :
            self._adjust_centroids(data)



def main():
    X_train = np.array([[1, 2],
                  [1.5, 1.8],
                  [5, 8 ],
                  [8, 8],
                  [1, 0.6],
                  [9,11]])

    plt.scatter(X_train[:,0], X_train[:,1])
    #plt.show()

    clf = clusteringClassifier(k_value=2,minkowski_p=2,max_iter=300)
    clf.fit(X_train)

    print clf.centroids_









if __name__ == "__main__":
    main()
