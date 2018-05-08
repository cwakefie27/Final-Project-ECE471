from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import performance

def run(X_train,y_train,X_test,y_test,predciction_filename=None,graph_name=None):
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

    if graph_name != None:
        performance.plot_roc(gs,X_test,y_test,graph_name)

    return performance.get_scores(gs.best_params_,predicted_classes,y_test,predciction_filename)
