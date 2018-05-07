from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import os
import sys

def get_results(gs,predicted_classes,y_test,predciction_filename):

    classifier = gs.best_params_;
    accuracy = accuracy_score(predicted_classes,y_test);
    confusion_matrix = get_confusion_matrix(predicted_classes,y_test)
    precision = precision_score(predicted_classes, y_test, average='macro')
    recall = recall_score(predicted_classes, y_test, average='macro')
    f1 = f1_score(predicted_classes, y_test, average='macro')

    if predciction_filename != None:
        # print ([[predicted,actual] for predicted,actual in zip(predicted_classes,y_test)])

        #Append results
        directory = "Predictions";
        if not os.path.exists(directory):
            os.makedirs(directory)

        string = "";
        with open(os.path.join(directory,predciction_filename), 'w') as file:
            string = predciction_filename;
            file.write(string+"\n");

    return accuracy,classifier,confusion_matrix,precision,recall,f1;
