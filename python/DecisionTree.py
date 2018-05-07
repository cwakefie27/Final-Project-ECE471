from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

def run(X_train,y_train,X_test,y_test,outputGraph=False,collapseType=-1):
    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid = {
                    'criterion':['gini','entropy'],
                    'max_depth': np.arange(2, 8)
                 }
    gs = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=4,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)

    classifier = gs.best_params_;
    predicted_classes = gs.best_estimator_.predict(X_test)
    accuracy = accuracy_score(predicted_classes,y_test);
    confusion_matrix = get_confusion_matrix(predicted_classes,y_test)
    precision = precision_score(predicted_classes, y_test, average='macro')
    recall = recall_score(predicted_classes, y_test, average='macro')
    f1 = f1_score(predicted_classes, y_test, average='macro')

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
            print ("ERROR: Collapse Type specified does not exists")
            return accuracy,classifier;

        feature_names = ['Age','Gender','Education','Country','Ethnicity','Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness','Impulsiveness','Sensation']
        tree_data = tree.export_graphviz(gs.best_estimator_, out_file=None,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,special_characters=True)
        graph = graphviz.Source(tree_data)
        graph.render("Decision_Tree")

    return accuracy,classifier,confusion_matrix,precision,recall,f1;
