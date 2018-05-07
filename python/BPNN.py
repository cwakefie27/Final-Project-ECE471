from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np

def run(X_train,y_train,X_test,y_test):
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

    classifier = gs.best_params_;
    predicted_classes = gs.best_estimator_.predict(X_test)
    accuracy = accuracy_score(predicted_classes,y_test);
    confusion_matrix = get_confusion_matrix(predicted_classes,y_test)
    precision = precision_score(predicted_classes, y_test, average='macro')
    recall = recall_score(predicted_classes, y_test, average='macro')
    f1 = f1_score(predicted_classes, y_test, average='macro')

    return accuracy,classifier,confusion_matrix,precision,recall,f1;
