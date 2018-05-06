from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
import numpy as np

def main():
    X_train = [[np.round(i,decimals=1),np.round(i,decimals=1)] for i in np.arange(0,3,.1)]
    y_train = [int(i) for i in np.arange(0,3,.1)]

    X_test = [[np.round(i,decimals=1)+.1,np.round(i,decimals=1)-.1] for i in np.arange(0,3,.1)]
    y_test = [int(i) for i in np.arange(0,3,.1)]

    feature_names = ['0','1']
    class_names = ['0','1','2']

    #Find the best parameters using GridSearchCV -- SPECIFY param_grid
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(3, 10)}
    gs = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=10,n_jobs=-1,scoring='accuracy')
    gs.fit(X_train,y_train)

    #Save Deceision tree
    tree_data = tree.export_graphviz(gs.best_estimator_, out_file=None,feature_names=feature_names, class_names=class_names, filled=True, rounded=True,special_characters=True)

    graph = graphviz.Source(tree_data)
    graph.render("Decision_Tree")

    print "Accuracy:",accuracy_score(gs.best_estimator_.predict(X_test),y_test)*100," Classifier:", gs.best_params_


if __name__ == "__main__":
    main()
