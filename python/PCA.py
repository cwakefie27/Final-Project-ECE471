import numpy as np
from sklearn.decomposition import PCA as PCA
import performance

def run(X_train,y_train,X_test,graph_name=None,n_components=-1):
    #Find optimal n components and retrain PCA or use specified
    if graph_name == None:
        PCAobject = PCA().fit(X_train)
        n_components = next(x[0] for x in enumerate(np.cumsum(PCAobject.explained_variance_ratio_)) if x[1] > 0.90) + 1
        PCAobject = PCA(n_components=n_components).fit(X_train)
    else:
        n_components = 2
        PCAobject = PCA(n_components=n_components).fit(X_train)

    #Transform datasets
    pca_train = PCAobject.transform(X_train)
    pca_test = PCAobject.transform(X_test)

    if graph_name != None:
        performance.plot_reduction(pca_train,y_train,graph_name)

    return pca_train,pca_test

if __name__ == "__main__":
    # Example usage
    X_train = [[i,i/2,i*5,i-1] for i in np.arange(0,3,.1)]
    pca_train,pca_test = run(X_train,X_train)
