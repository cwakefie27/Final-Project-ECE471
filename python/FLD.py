from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

def run(X_train,y_train,X_test):
    #Find optimal n components and retrain FLD
    lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
    n_components = next(x[0] for x in enumerate(np.cumsum(lda.explained_variance_ratio_)) if x[1] > 0.90) + 1
    lda = LinearDiscriminantAnalysis(n_components=n_components).fit(X_train, y_train)
    #Transform datasets
    fld_train = lda.transform(X_train)
    fld_test = lda.transform(X_test)
    return fld_train,fld_test

if __name__ == "__main__":
    # Example usage

    from sklearn import datasets
    from sklearn.datasets import load_files
    import matplotlib.pyplot as plt

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names

    X_r2,duplicate = run(X,y,X)

    #if the data is 2D we can graph it
    if len(X_r2[0]) == 2:
        colors = ['navy', 'turquoise', 'darkorange']
        lw = 2
        plt.figure()
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                        label=target_name)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title('LDA of IRIS dataset')
        plt.show()
    else:
        print ("New Dimensions: " + str(len(X_r2[0])))
