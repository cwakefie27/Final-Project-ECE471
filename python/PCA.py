import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as PCA

def run(X_train,X_test):
    #Find optimal n components and retrain PCA
    PCAobject = PCA().fit(X_train)
    n_components = next(x[0] for x in enumerate(np.cumsum(PCAobject.explained_variance_ratio_)) if x[1] > 0.90) + 1
    PCAobject = PCA(n_components=n_components).fit(X_train)
    #Transform datasets
    pca_train = PCAobject.transform(X_train)
    pca_test = PCAobject.transform(X_test)

    return pca_train,pca_test

if __name__ == "__main__":
    # Example usage
    X_train = [[i,i/2,i*5,i-1] for i in np.arange(0,3,.1)]
    X_project = [[i,i,i,i] for i in np.arange(0,3,.1)]
    pca_train,pca_test = run(X_train,X_project)
    print pca_train
