from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
import performance

def run(X_train,y_train,X_test,y_test,predciction_filename=None,graph_name=None):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid ={
                    'C': [1,2,3,4,5,6,7],
                    'gamma': [0.001,0.0001,0.01,0.1,1,10,100],
                    'probability': [True]
                }
    gs = GridSearchCV(SVC(), param_grid, cv=2,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)
    predicted_classes = gs.best_estimator_.predict(X_test)

    if graph_name != None:
        performance.plot_roc(gs,X_test,y_test,graph_name)

    return performance.get_scores(gs.best_params_,predicted_classes,y_test,predciction_filename)
