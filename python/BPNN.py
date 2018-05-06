from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np

def run(X_train,y_train,X_test,y_test):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid ={
                    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                    'solver' : ['adam','lbfgs'],
                    'alpha' : [0.0001],
                    'batch_size': ['auto'],
                    'learning_rate_init': [0.001],
                    'max_iter' : [2000], #Higher numbers will avoid ConvergenceWarnings
                    'hidden_layer_sizes': [ (1,2),(1,),(2,)]
                }
    gs = GridSearchCV(MLPClassifier(), param_grid, cv=2,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)

    accuracy = accuracy_score(gs.best_estimator_.predict(X_test),y_test)*100;
    classifier = gs.best_params_;

    return accuracy,classifier;