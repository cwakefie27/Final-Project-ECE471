import numpy as np
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class clusteringClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, algo='kMeans', k_value=2, epsilon=.1, minkowski_p=2, max_iter=300):
        self.algo = algo
        self.k_value = k_value
        self.epsilon = epsilon
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
            #Find the closest centroid
            centroid_assignment_index = np.argmin([self._minkowski_distance(data[i],self.centroids_[j]) for j in range(0,len(self.centroids_))])

            if (self.data_centroid_indexes_[i] != centroid_assignment_index):
                centroids_changed = True
                self.data_centroid_indexes_[i] = centroid_assignment_index;

        return centroids_changed

    def _adjust_centroids_kMeans(self,data):
        self.centroids_[:] = 0
        count = np.zeros(len(self.centroids_))
        for data_index,centroid_index in enumerate(self.data_centroid_indexes_):
            self.centroids_[centroid_index] += data[data_index]
            count[centroid_index] += 1
        #Divide by the count to adjust centroids
        self.centroids_ = self.centroids_/count.reshape((len(count),1))

    def _adjust_centroids_WTA(self,data):
        for i in range(0,len(data)):
            w_old = self.centroids_[self.data_centroid_indexes_[i]];
            self.centroids_[self.data_centroid_indexes_[i]] = w_old + (data[i] - w_old) * self.epsilon;

    def fit(self,X,y):
        assert self.k_value <= len(X), "Number of centroids is greater than number of instances in data"

        #Assign Centroids random values in data
        self.centroids_ = np.array([np.array(X[i]) for i in random.sample(range(len(X)), self.k_value)])
        self.data_centroid_indexes_ = [0] * len(X)

        iterations = 0

        if self.algo == 'kMeans':
            while self._assign_to_centroids(X) == True and iterations < self.max_iter:
                self.epsilon = -1
                self._adjust_centroids_kMeans(X)
                iterations += 1
        elif self.algo == 'WTA':
            while iterations < self.max_iter:
                self._assign_to_centroids(X)
                self._adjust_centroids_WTA(X)
                iterations += 1

        #associate each centroid with a class by voting for the most frequent class in each centroid
        unique_classes = np.unique(y)
        class_votes = np.zeros((len(self.centroids_),len(unique_classes)),dtype=int)
        for index,class_value in zip(self.data_centroid_indexes_,y):
            class_votes[index][np.where(unique_classes==class_value)[0][0]] += 1

        self.centroid_classes_ = [unique_classes[np.argmax(vote)] for vote in class_votes]

    def _predict_isntance(self, point):
        #find the closest centroid then use it to key into the centroids class
        closest_centroid = np.argmin([self._minkowski_distance(point,self.centroids_[i]) for i in range(0,len(self.centroids_))])
        return self.centroid_classes_[closest_centroid]

    #predict all
    def predict(self, X):
        try:
            getattr(self, "centroid_classes_")
        except AttributeError:
            raise RuntimeError("Fit the classifier before using it")

        return([self._predict_isntance(x) for x in X])

    def score(self, X, y):
        try:
            getattr(self, "centroid_classes_")
        except AttributeError:
            raise RuntimeError("Must know the classes of the data in order to score each configuration")

        predicted_X = self.predict(X)
        #print predicted_X,y
        return accuracy_score(predicted_X,y)

def run(X_train,y_train,X_test,y_test):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    #epsilon is only used for WTA
    param_grid = [
            {
                'algo':['kMeans'],
                'k_value': [2],
                'minkowski_p':[1,2,np.inf],
                'max_iter': [10]
            },
            {
                'algo':['kMeans','WTA'],
                'epsilon': [.00001],
                'k_value': [2],
                'minkowski_p':[1,2,np.inf],
                'max_iter': [10]
            }
                 ]
    gs = GridSearchCV(clusteringClassifier(), param_grid, cv=2,n_jobs=-1)
    gs.fit(X_train,y_train)

    classifier = gs.best_params_;
    predicted_classes = gs.best_estimator_.predict(X_test)
    accuracy = accuracy_score(predicted_classes,y_test);
    confusion_matrix = get_confusion_matrix(predicted_classes,y_test)
    precision = precision_score(predicted_classes, y_test, average='macro')
    recall = recall_score(predicted_classes, y_test, average='macro')
    f1 = f1_score(predicted_classes, y_test, average='macro')

    return accuracy,classifier,confusion_matrix,precision,recall,f1;
