from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as get_confusion_matrix
import numpy as np
import sys

class MPPClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, case=1):
        self.case = case

    def fit(self, X, y, priors=None):
        X,y = np.array(X),np.array(y)
        # split the data based on class, calculate important info
        data         = {i: X[np.where(y==[i])] for i in np.unique(y)}
        self.means_  = {i: np.mean(data[i].T,axis=1) for i in data.keys()}
        self.covs_   = {i: np.cov(data[i].T) for i in data.keys()}
        self.covavg_ = np.sum([self.covs_[i] for i in data.keys()],axis=0) / len(data.keys())
        self.varavg_ = np.sum(self.covavg_.diagonal()[:len(data.keys())]) / len(data.keys())
        #set equal priors if not specified
        if priors != None:
            assert (len(priors) == len(data.keys())), "ERROR: Priors dimension is not the same as number of classes"
            assert (sum(priors) == 1) , "ERROR: Sum of priors does not equal 1s"
            self.priors_ = priors
            self.priors_ = {key: priors[i] for i,key in enumerate(data.keys())}
        else:
            self.priors_ = {key: 1.0/len(data.keys()) for key in data.keys()}
        return self

    def _minkowski_distance(self,u,v,minkowski_p):
        if minkowski_p == np.inf:
            return float(max(np.abs(u-v)))
        else:
            return sum(abs(u-v)**(float(minkowski_p))) ** (1/float(minkowski_p))

    def _mah_distance(self,point,cov,mean):
        differnce = point - mean
        try:
            squared_distance = differnce.T.dot(np.linalg.inv(cov)).dot(differnce)
            if squared_distance > 0:
                return np.sqrt(squared_distance)
            else:
                return np.inf
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                return np.inf
            else:
                raise

    def _predict_isntance(self, point):
        distances = {i: 0 for i in self.means_.keys()}
        if self.case == 1:
            for i in self.means_.keys():
                edist = self._minkowski_distance(point, self.means_[i],2);
                distances[i] = -edist * edist / (2* self.varavg_) + np.log(self.priors_[i])
        if self.case == 2:
            for i in self.means_.keys():
                mdist = self._mah_distance(point,self.covavg_,self.means_[i])
                distances[i] = -0.5 * mdist * mdist + np.log(self.priors_[i])
        if self.case == 3:
            for i in self.means_.keys():
                mdist = self._mah_distance(point, self.covs_[i], self.means_[i])
                det_value = np.linalg.det(self.covs_[i])
                #Avoid singular covariance matrices
                if mdist == np.inf or det_value <= 0:
                    distances[i] = np.inf
                else:
                    distances[i] = -0.5 * mdist * mdist - 0.5 * np.log(np.linalg.det(self.covs_[i])) + np.log(self.priors_[i])
        return max(distances, key=distances.get)

    #predict all
    def predict(self, X):
        try:
            getattr(self, "means_")
        except AttributeError:
            raise RuntimeError("Fit the classifier before using it")

        return([self._predict_isntance(x) for x in X])

    def score(self, X, y):
        predicted_X = self.predict(X)
        return accuracy_score(predicted_X,y)*100

def run(X_train,y_train,X_test,y_test):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid = {
                    'case':[1,2,3],
                 }
    gs = GridSearchCV(MPPClassifier(), param_grid,cv=2,n_jobs=1)
    gs.fit(X_train,y_train)

    classifier = gs.best_params_;
    predicted_classes = gs.best_estimator_.predict(X_test)
    accuracy = accuracy_score(predicted_classes,y_test)*100;
    confusion_matrix = get_confusion_matrix(predicted_classes,y_test)

    return accuracy,classifier,confusion_matrix;
