from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
import numpy as np

def run(X_train,y_train,X_test,y_test,outputGraph):
    #USED FOR GRAPH
    feature_names = ['0','1']
    class_names = ['0','1','2']

    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid = {
                    'criterion':['gini','entropy'],
                    'max_depth': np.arange(3, 10)
                 }
    gs = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=10,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)

    #Save Deceision tree
    if outputGraph == True:
        tree_data = tree.export_graphviz(gs.best_estimator_, out_file=None,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,special_characters=True)
        graph = graphviz.Source(tree_data)
        graph.render("Decision_Tree")

    accuracy = accuracy_score(gs.best_estimator_.predict(X_test),y_test)*100;
    classifier = gs.best_params_;

    return accuracy,classifier;