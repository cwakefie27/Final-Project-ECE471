from sklearn.model_selection import GridSearchCV
import numpy as np
import performance
from sklearn.ensemble import RandomForestClassifier

def run(X_train,y_train,X_test,y_test,predciction_filename=None,graph_name=None):
    param_grid ={
                    'n_estimators':[10,20],
                    'max_depth':[2],
                    'criterion':['gini','entropy'],
                    'max_depth':[3,4,5,6]
                }

    gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=2,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)
    predicted_classes = gs.best_estimator_.predict(X_test)

    if graph_name != None:
        performance.plot_roc(gs,X_test,y_test,graph_name)

    return performance.get_scores(gs.best_params_,predicted_classes,y_test,predciction_filename)
