from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import numpy as np
import performance

def run(X_train,y_train,X_test,y_test,predciction_filename=None):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid ={
                    'activation' : ['identity', 'logistic', 'tanh'],
                    'alpha' : [0.0001],
                    'batch_size': ['auto'],
                    'learning_rate_init': [0.001],
                    'max_iter' : [2000], #Higher numbers will avoid ConvergenceWarnings
                    'hidden_layer_sizes': [ (1,),(2,),(5,),(2,2),(3,5)]
                }
    gs = GridSearchCV(MLPClassifier(), param_grid, cv=2,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)
    predicted_classes = gs.best_estimator_.predict(X_test)

    # calculate the fpr and tpr for all thresholds of the classification
    probs = gs.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    return performance.get_results(gs,predicted_classes,y_test,predciction_filename)
