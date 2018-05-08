import pandas as pd
import numpy as np
import sys as sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os

START_SPACE = "     "

def get_class_labels(collapseType):
    if collapseType == 0:
        class_names = ['Never used','Used over a decade ago','Used in last decade','Used in last year','Used in last month','Used in last week','Used in last day']
    elif collapseType == 1:
        class_names = ['Never used','Used at somepoint']
    elif collapseType == 2:
        class_names = ['Never used','Used over a decade ago','Used within the decade']
    elif collapseType == 3:
        class_names = ['Never used','Used over a year ago','Used within the year']
    else:
        print ("ERROR: Collapse Type specified does not exists in fusion")
        sys.exit()
    return class_names

#Build a confusion matrix from prediction list
def build_confusion(prediction_list,classes):
    #unique classes key into the confusion matrix
    unique_classes = np.unique(classes)
    confusion_matrix = np.zeros((len(unique_classes),len(unique_classes)),dtype=int)
    for prediction in prediction_list:
        #Convert the class label to a possible index in confusion confusion_matrix
        predicted_class = np.where(unique_classes == prediction[0])[0][0]
        actual_class = np.where(unique_classes == prediction[1])[0][0]
        #increment confusion matrix count
        confusion_matrix[actual_class][predicted_class] += 1
    return confusion_matrix,unique_classes

def print_confusion(confusion_matrix,class_labels):
    #Get the size of the largest actual label and add 2 to get predicted size
    actual_length =  np.array([len('Actual ' + str(label)) for label in class_labels]).max()
    predicted_length = actual_length + 3
    #ensure the predicted length is long enough to accomadate percent correct
    if predicted_length < 17:
        predicted_length = 17

    #print predicted labels
    print (START_SPACE + '{:{actual_width}}|'.format('',actual_width=actual_length+3)),
    for label in class_labels:
        print (' {:>{predicted_width}} |'.format('Predicted ' + str(label),predicted_width=predicted_length)),
    print ("")

    #print confusion matrix
    for row_index,row in enumerate(zip(confusion_matrix,class_labels)):
        #print actual label
        print (START_SPACE + '| {:{actual_width}} |'.format('Actual ' + str(row[1]),actual_width=actual_length)),
        #print each instance in the confusion matrix
        for col_index,instance in enumerate(row[0]):
            print (' {:{predicted_width}} |'.format(int(instance),predicted_width=predicted_length)),
        #print the percent correct per row aka actual percent correct
        print ('{:.3f} % correct'.format((row[0][row_index] / float(sum(row[0])))*100))

    #print the percent correct per column aka predicted percent correct
    print (START_SPACE + '{:{actual_width}}|'.format('',actual_width=actual_length+3)),
    for col_index in range(0,len(confusion_matrix)):
        print (' {:>{predicted_width}.3f} % correct |'.format((confusion_matrix[col_index][col_index] / max(1,float(sum(confusion_matrix[:,col_index]))))*100,predicted_width=int(predicted_length-10))),
    print ("")

def print_stats(predicted_list,actual_list):
    accuracy = accuracy_score(predicted_list,actual_list);
    precision = precision_score(predicted_list, actual_list, average='macro')
    recall = recall_score(predicted_list, actual_list, average='macro')
    f1 = f1_score(predicted_list, actual_list, average='macro')

    print (START_SPACE + "Accuracy  : " + str(np.round(accuracy*100,decimals=3)))
    print (START_SPACE + "Precision : " + str(np.round(precision*100,decimals=3)))
    print (START_SPACE + "Recall    : " + str(np.round(recall*100,decimals=3)))
    print (START_SPACE + "F1 score  : " + str(np.round(f1*100,decimals=3)))

def fuse_classifiers(first_classifier_percent,second_classifier_percent,all_classes):
    #Fuse the classifiers and create a lookup table
    lookup_table = np.zeros((len(all_classes),len(all_classes)),dtype=int)
    for i,col_first in enumerate(first_classifier_percent.T):
        for j,col_second in enumerate(second_classifier_percent.T):
            lookup_table[i][j] = np.argmax(col_first * col_second)
    return lookup_table

def predict_all_fusion(first_predictions,second_predictions,actual_classes,lookup_table):
    prediction_list = []
    lookup_key = np.unique(actual_classes)
    for predictions in zip(first_predictions,second_predictions,actual_classes):
        #predict using lookup table, then translate back to correct label
        first_prediction = np.where(lookup_key==predictions[0])[0][0]
        second_prediction = np.where(lookup_key==predictions[1])[0][0]
        prediction_list.append([lookup_key[lookup_table[first_prediction][second_prediction]],predictions[2]])
    return np.array(prediction_list)

#Divide all rows of confusion matrix by sum of row to get percent likelihood
def divide_matrix_by_rows(confusion_matrix):
    row_sum = np.sum(confusion_matrix,axis=1).reshape(len(confusion_matrix),1).astype(float)
    return  confusion_matrix / row_sum

def main():
    #Get all parameters of program
    try:
        FIRST_PREDICTIONS = sys.argv[1]
        SECOND_PREDICTIONS  = sys.argv[2]
    except IndexError as e:
        print ("USAGE: python fusion.py first_predictions.csv second_predictions.csv")
        sys.exit()

    #Read file including the header
    with open(FIRST_PREDICTIONS,'r') as file:
        first_header = file.readline().strip()
        first_prediction_list = np.array(pd.read_csv(file, error_bad_lines=False, header=None,comment='#'))
    file.closed

    with open(SECOND_PREDICTIONS,'r') as file:
        second_header = file.readline().strip()
        second_prediction_list = np.array(pd.read_csv(file, error_bad_lines=False, header=None,comment='#'))
    file.closed

    COLLAPSE_TYPE = int(os.path.basename(FIRST_PREDICTIONS).split('_')[3])
    assert (COLLAPSE_TYPE == int(os.path.basename(SECOND_PREDICTIONS).split('_')[3])), "Collapse types must be the same"

    #Get all classes in both files and the actual classes
    all_classes = np.unique(np.concatenate([first_prediction_list,second_prediction_list]))
    actual_classes = first_prediction_list[:,1]
    class_labels = get_class_labels(COLLAPSE_TYPE)

    # #create confusion matrices from the predictions
    first_confusion_matrix,first_class_labels = build_confusion(first_prediction_list,all_classes)
    second_confusion_matrix,second_class_labels = build_confusion(second_prediction_list,all_classes)

    #print info on first and second set of predictions
    print ("*** First Classifier Information ***")
    print (first_header)
    print_stats(first_prediction_list[:,0],first_prediction_list[:,1])
    print ("")
    print_confusion(first_confusion_matrix,class_labels)
    print ("")
    print ("*** Second Classifier Information *** ")
    print (second_header)
    print_stats(second_prediction_list[:,0],first_prediction_list[:,1])
    print ("")
    print_confusion(second_confusion_matrix,class_labels)
    print ("")

    #Get the percent likelihood of each index being picked
    first_percent_likelihood_matrix  = divide_matrix_by_rows(first_confusion_matrix)
    second_percent_likelihood_matrix = divide_matrix_by_rows(second_confusion_matrix)

    #create a lookup table by fusing the classifiers
    lookup_table = fuse_classifiers(first_percent_likelihood_matrix,second_percent_likelihood_matrix,all_classes)

    #predict all using the lookup table and build confusion matrix
    fused_prediction_list = predict_all_fusion(first_prediction_list[:,0],second_prediction_list[:,0],actual_classes,lookup_table)
    fused_confusion_matrix,fused_class_labels = build_confusion(fused_prediction_list,all_classes)

    print ("*** Fused Classifier Information ***")
    print_stats(fused_prediction_list[:,0],fused_prediction_list[:,1])
    print
    print_confusion(fused_confusion_matrix,class_labels)
    print

if __name__ == "__main__":
	main()
