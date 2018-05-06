import numpy as np
import matplotlib.pyplot as plt

# Calculate the covaraince for a dataset
def Cov(data):
    return np.dot((data - data.mean()).T , (data - data.mean())) / (data.shape[0]-1)

# Returns Descending order Eigen Values paired with their eigenVector
def EigenPairs(eig_vals, eig_vecs):
    # Pair vectors and values together to be sorted in Descending order
    eig_pairs = [{'eig_val':np.abs(eig_vals[i]), 'eig_vec':eig_vecs[:,i]} for i in range(len(eig_vals))]
    return sorted(eig_pairs, key=lambda k: k['eig_val'],reverse=True)

# Build the projection matrix
def ComposeProjectionMatrix(eig_pairs, n_components):
    n_features = len(eig_pairs)
    if n_components <= 0:
        print ("Zero componets requested in PCA")
        sys.exit()
    if n_components >= n_features+1:
        print ("Number of componets requested in PCA is greater than number of features")
        sys.exit()
    matrix_w = eig_pairs[0]['eig_vec'].reshape(n_features,1)
    for i in range(1,n_components):
        matrix_w = np.hstack((matrix_w,eig_pairs[i]['eig_vec'].reshape(n_features,1)))
    return matrix_w

def PCA(train_data,data_to_transform,n_components=-1):
    train_data = np.array(train_data)

    #Center the data about the training mean
    train_mean = train_data.mean(axis=0)
    train_data = train_data - train_mean
    data_to_transform = data_to_transform - train_mean

    # Calculate Covaraince Matrix for the training dataset
    cov_train = Cov(train_data)
    # Calculate the Eiegen values and vectors
    eig_vals, eig_vecs = np.linalg.eig(cov_train)
    eig_sum = np.sum(eig_vals)

    variance_explained = [(i / sum(eig_vals)) for i in sorted(eig_vals, reverse=True)]
    cumulative_variance_explained = np.cumsum(variance_explained)
    for i,cum_var_expl in enumerate(cumulative_variance_explained):
        if cum_var_expl >= .9 and n_components == -1:
            #increase by one to account for previously added component
            n_components = i + 1
            break

    eig_pairs = EigenPairs(eig_vals, eig_vecs )
    projection_matrix = ComposeProjectionMatrix(eig_pairs,n_components)
    transformed_data = np.dot(data_to_transform,projection_matrix)

    return transformed_data

def main():
    X_train = [[np.round(i,decimals=1),np.round(i,decimals=1)] for i in np.arange(0,3,.1)]
    X_project = [[np.round(i,decimals=1)+.1,np.round(i,decimals=1)-.1] for i in np.arange(0,3,.1)]

    #Perform PCA
    transformed_data = PCA(X_train,X_project,2)
    print transformed_data

    #SKLEARN PCA
    # from sklearn.decomposition import PCA as sklearnPCA
    # sklearn_pca = sklearnPCA(n_components=2).fit(X_train)
    # transformed_data = sklearn_pca.transform(X_project)
    # print sklearn_pca.components_
    # print transformed_data

    #PLOT2D TEST
    # plt.figure("PCA PC1 vs Class")
    # plt.xlabel('PC1', fontsize=10)
    # plt.ylabel('Class', fontsize=10)
    # plt.scatter(transformed_data[:,0],transformed_data[:,1], color="red", alpha=0.2,label="Class 0")
    # plt.legend(loc='upper center',bbox_to_anchor=(.5, 1.125), ncol=2, fancybox=True, shadow=True)
    # plt.show()

if __name__ == "__main__":
    main()
