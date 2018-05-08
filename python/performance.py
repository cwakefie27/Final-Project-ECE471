from sklearn.metrics import confusion_matrix as get_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def get_scores(classifier,predicted_classes,y_test,predciction_filename):
    accuracy = accuracy_score(predicted_classes,y_test);
    confusion_matrix = get_confusion_matrix(predicted_classes,y_test)
    precision = precision_score(predicted_classes, y_test, average='macro')
    recall = recall_score(predicted_classes, y_test, average='macro')
    f1 = f1_score(predicted_classes, y_test, average='macro')

    if predciction_filename != None:
        prediction_list = [[predicted,actual] for predicted,actual in zip(predicted_classes,y_test)]

        #Append results
        directory = "Predictions";
        if not os.path.exists(directory):
            os.makedirs(directory)

        string = "";
        with open(os.path.join(directory,predciction_filename), 'w') as file:
            string = predciction_filename;
            file.write(string+"\n");

        header_string = predciction_filename
        np.savetxt(os.path.join(directory,predciction_filename),prediction_list,'%s',header=header_string,delimiter = ",")

    return classifier,accuracy,precision,recall,f1,confusion_matrix

def plot_reduction(data,class_values,filename):
    if len(np.unique(class_values)) > 7:
        print ("ERROR: Colors and Markers do not support this many classes")
        return

    colors = ['red','green','blue','cyan','purple','chocolate','gray']
    markers = ['o', 'v',    '*',   's',     '*',     'h',   '+']
    class_labels = get_class_labels(int(os.path.basename(filename).split('_')[3]))

    colors_classes = [colors[class_value] for class_value in class_values]
    markers_classes = [markers[class_value] for class_value in class_values]

    if len(data[0]) == 2:
        plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

        for class_value in np.unique(class_values):
            class_data = data[class_value==class_values]
            plt.scatter(class_data[:,0],class_data[:,1],color=colors[class_value],marker=markers[class_value],label=class_labels[class_value],s=50,alpha=.4)

        plt.title(get_graph_name(filename))
        plt.xlabel("PC1", fontsize=12)
        plt.ylabel("PC2", fontsize=12)
        plt.legend(ncol=10,bbox_to_anchor=[0.5, -0.135], loc='lower center')
    elif len(data[0]) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(num=None, figsize=(15, 9), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, projection='3d')

        for class_value in np.unique(class_values):
            class_data = data[class_value==class_values]
            ax.scatter(class_data[:,0],class_data[:,1],class_data[:,2],color=colors[class_value],marker=markers[class_value],label=class_labels[class_value],s=50,alpha=.3)

        plt.title(get_graph_name(filename))
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend(ncol=10,bbox_to_anchor=[0.5, -0.135], loc='lower center')
    else:
        print ("\nERROR: Cannot graph reduction because n_components it not 2 or 3\n")
        return
    plt.show()


def plot_roc(gs,X_test,y_test,filename):
    if len(np.unique(y_test)) != 2:
        print ("\nERROR: Cannot graph ROC curve because number of classes is not 2\n")
        return
    probs = gs.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(filename)
    plt.title(get_graph_name(filename))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

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
        print ("ERROR: Collapse Type specified does not exists in performance")
        sys.exit()
    return class_names

def get_graph_name(filename):
    params = os.path.splitext(os.path.basename(filename))[0].split('_')
    graph_name = params[0].title() + "  "
    graph_name += params[2] + " "
    graph_name += "Reduction: " + params[1] + "  "
    graph_name += "Collapse Type: " + params[3] + "  "
    graph_name += "Cols: " + params[4] + "  "
    return graph_name

START_SPACE = "     "

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
        print ('{:.3f} % correct'.format((row[0][row_index] / max(1,float(sum(row[0]))))*100))

    #print the percent correct per column aka predicted percent correct
    print (START_SPACE + '{:{actual_width}}|'.format('',actual_width=actual_length+3)),
    for col_index in range(0,len(confusion_matrix)):
        print (' {:>{predicted_width}.3f} % correct |'.format((confusion_matrix[col_index][col_index] / max(1,float(sum(confusion_matrix[:,col_index]))))*100,predicted_width=int(predicted_length-10))),
    print ("")
