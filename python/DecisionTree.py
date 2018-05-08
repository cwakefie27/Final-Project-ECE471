from sklearn.model_selection import GridSearchCV
from sklearn import tree
import graphviz
import numpy as np
import performance
import sys

def run(X_train,y_train,X_test,y_test,collapseType=-1,predciction_filename=None,graph_name=None):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid = {
                    'criterion':['gini','entropy'],
                    'max_depth': np.arange(2, 8)
                 }
    gs = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=4,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)
    predicted_classes = gs.best_estimator_.predict(X_test)

    classifier,accuracy,precision,recall,f1,confusion_matrix = performance.get_scores(gs.best_params_,predicted_classes,y_test,predciction_filename)

    #Save Deceision tree
    if graph_name != None:
    	class_names = performance.get_class_labels(collapseType)
        feature_names = ['Age','Gender','Education','Country','Ethnicity','Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness','Impulsiveness','Sensation']
        tree_data = tree.export_graphviz(gs.best_estimator_, out_file=None,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,special_characters=True)
        graph = graphviz.Source(tree_data)
        graph.render(graph_name+"_DT")

        #predciction_filename is not the rite naming scheme but it should work for these purposes
        performance.plot_roc(gs,X_test,y_test,predciction_filename)


    return classifier,accuracy,precision,recall,f1,confusion_matrix
