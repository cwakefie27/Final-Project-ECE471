import pandas as pd
import numpy as np
import sys as sys

# CLASS_LABELS_PIMA = ["Negative","Positive"]
# CLASS_LABELS_FGLASS = ["WinF","WinNF","Veh","Con","Tabl","Head"]

#Get all parameters of program
try:
    FIRST_PREDICTIONS = sys.argv[1]
    SECOND_PREDICTIONS  = sys.argv[2]
except IndexError as e:
    print ("USAGE: python fusion.py first_predictions.csv second_predictions.csv")
    sys.exit()

#Read file including the header
with open(FIRST_PREDICTIONS,'r') as file:
    # first_header = file.readline().strip()
    first_prediction_list = np.array(pd.read_csv(file, error_bad_lines=False, header=None,comment='#'))
file.closed

with open(SECOND_PREDICTIONS,'r') as file:
    # second_header = file.readline().strip()
    second_prediction_list = np.array(pd.read_csv(file, error_bad_lines=False, header=None,comment='#'))
file.closed

#Get all classes in both files and the actual classes
all_classes = np.unique(np.concatenate([first_prediction_list,second_prediction_list]))
actual_classes = first_prediction_list[:,1]

# #create confusion matrices from the predictions
# first_confusion_matrix,first_class_labels = kNN.build_confusion(first_prediction_list,all_classes)
# second_confusion_matrix,second_class_labels = kNN.build_confusion(second_prediction_list,all_classes)

#print info on first and second set of predictions
print ("*** First Classifier Information ***")
print (first_header)
kNN.print_stats_2x2(first_confusion_matrix)
print
kNN.print_confusion(first_confusion_matrix,CLASS_LABELS_PIMA)
print
print ("*** Second Classifier Information *** ")
print (second_header)
kNN.print_stats_2x2(second_confusion_matrix)
print
kNN.print_confusion(second_confusion_matrix,CLASS_LABELS_PIMA)
print

#Divide all rows of confusion matrix by sum of row to get percent likelihood
def divide_matrix_by_rows(confusion_matrix):
    row_sum = np.sum(confusion_matrix,axis=1).reshape(len(confusion_matrix),1).astype(float)
    return  confusion_matrix / row_sum

#Get the percent likelihood of each index being picked
first_percent_likelihood_matrix  = divide_matrix_by_rows(first_confusion_matrix)
second_percent_likelihood_matrix = divide_matrix_by_rows(second_confusion_matrix)

def fuse_classifiers(first_classifier_percent,second_classifier_percent,num_classes):
    #Fuse the classifiers and create a lookup table
    lookup_table = np.zeros((len(all_classes),len(all_classes)),dtype=int)
    for i,col_first in enumerate(first_classifier_percent.T):
        for j,col_second in enumerate(second_classifier_percent.T):
            lookup_table[i][j] = np.argmax(col_first * col_second)
    return lookup_table

#create a lookup table by fusing the classifiers
lookup_table = fuse_classifiers(first_percent_likelihood_matrix,second_percent_likelihood_matrix,len(all_classes))

def predict_all_fusion(first_predictions,second_predictions,actual_classes,lookup_table):
    prediction_list = []
    lookup_key = np.unique(actual_classes)
    for predictions in zip(first_predictions,second_predictions,actual_classes):
        #predict using lookup table, then translate back to correct label
        first_prediction = np.where(lookup_key==predictions[0])[0][0]
        second_prediction = np.where(lookup_key==predictions[1])[0][0]
        prediction_list.append([lookup_key[lookup_table[first_prediction][second_prediction]],predictions[2]])
    return prediction_list

#predict all using the lookup table and build confusion matrix
fused_prediction_list = predict_all_fusion(first_prediction_list[:,0],second_prediction_list[:,0],actual_classes,lookup_table)
fused_confusion_matrix,fused_class_labels = kNN.build_confusion(fused_prediction_list,all_classes)

print ("*** Fused Classifier Information ***")
kNN.print_stats_2x2(fused_confusion_matrix)
print
kNN.print_confusion(fused_confusion_matrix,CLASS_LABELS_PIMA)
print
