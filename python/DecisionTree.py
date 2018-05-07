from sklearn.model_selection import GridSearchCV
from sklearn import tree
import graphviz
import numpy as np
import performance
import sys

def run(X_train,y_train,X_test,y_test,outputGraph=False,collapseType=-1,predciction_filename=None):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid = {
                    'criterion':['gini','entropy'],
                    'max_depth': np.arange(2, 8)
                 }
    gs = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=4,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)
    predicted_classes = gs.best_estimator_.predict(X_test)

    accuracy,classifier,confusion_matrix,precision,recall,f1 = performance.get_results(gs,predicted_classes,y_test,predciction_filename)

    #Save Deceision tree
    if outputGraph == True:
    	if collapseType == 0:
			class_names = ['Never used','Used over a decade ago','Used in last decade','Used in last year','Used in last month','Used in last week','Used in last day']
        elif collapseType == 1:
			class_names = ['Never used','Used at somepoint']
        elif collapseType == 2:
			class_names = ['Never used','Used over a decade ago','Used within the decade']
    	elif collapseType == 3:
			class_names = ['Never used','Used over a year ago','Used within the year']
        else:
            print ("ERROR: Collapse Type specified does not exists in DT")
            sys.exit()

        feature_names = ['Age','Gender','Education','Country','Ethnicity','Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness','Impulsiveness','Sensation']
        tree_data = tree.export_graphviz(gs.best_estimator_, out_file=None,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,special_characters=True)
        graph = graphviz.Source(tree_data)
        graph.render("Decision_Tree")

    return accuracy,classifier,confusion_matrix,precision,recall,f1;
