from __future__ import print_function
import numpy as np
import random
from datetime import datetime
import sys
import clustering
import DecisionTree
import kNN
import MPP
import BPNN
import MPP
import PCA
import FLD
import csv
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

#Print out information on how to run the program
def printHelp():
    print("\nPlease run the program with the following arguments: runner.py reduction_name ALGORITHIM_NAME DATASET_FILENAME COLLAPSE_TYPE COLUMNS_TO_USE");
    print("\tReductions to Use             : None PCA FLD");
    print("\tAlgorithims to Use            : Clustering, DecisionTree, KNN, BPNN, MPP");
    print("\tDataset Location              : Data_Drug_Consumption/sub_data/");
    print("\tClassification Collapse Types : 0 1 2 3");
    print("\t\t0: None, use original classes");
    print("\t\t1: Collapse to 2 classes. (0: Never used, 1: Used at somepoint)");
    print("\t\t2: Collapse to 3 classes. (0: Never used, 1: Used over a decade ago, 2: Used within the decade)");
    print("\t\t3: Collapse to 3 classes. (0: Never used, 1: Used over a year ago, 2: Used within the year)");
    print("\tColumns To Use                : 0-11 -1")
    print("\t\tPlease provide a comma seperated string of column indices to use. NO SPACES! ex: 0,3,4 to use the columns 0, 3, and 4.")
    print("\t\tPut -1 to use all columns")
    print("\t(Optional) Save predicitons?  : T or F")

#Split is about what percentage should go to the training set vs testing set
def loadDataset(filename, split, cols, X_train=[], y_train=[], X_test=[], y_test=[]):
    with open(filename,"rt") as csvfile:
        reader = csv.reader(csvfile);
        for row in reader:
            lin = [];
            for col in row:
                lin.append(float(col));

            line = []
            #Need to remove some columns
            if cols[0] != -1:
                for i in range(len(lin)):
                    found = False;
                    for j in range(len(cols)):
                        if cols[j] == i:
                            found = True;
                            break;
                    #Keep this column
                    if found == True:
                        line.append(lin[i]);
            else:
                line = lin[:-1];

            #Seperate into training and testing based on split variable
            if random.random() < split:
                X_train.append(line)
                y_train.append(lin[-1])
            else:
                X_test.append(line)
                y_test.append(lin[-1])

#Collapse the classifications from 7 classes to a more reasonable number
def collapseClassifications(_type,y_train=[],y_test=[]):
	#Collapse to 2 classifications => (0: Never used, 1: Used at somepoint)
	if _type == 1:
		for i in range(len(y_train)):
			if y_train[i] > 0:
				y_train[i] = 1;
			else:
				y_train[i] = 0;
		for i in range(len(y_test)):
			if y_test[i] > 0:
				y_test[i] = 1;
			else:
				y_test[i] = 0;
	#Collapse to 3 classifications => (0: Never used, 1: Used over a decade ago, 2: Used within the decade)
	elif _type == 2:
		for i in range(len(y_train)):
			if y_train[i] >= 2:
				y_train[i] = 2;
			elif y_train[i] >= 1:
				y_train[i] = 1;
			else:
				y_train[i] = 0;
		for i in range(len(y_test)):
			if y_test[i] >= 2:
				y_test[i] = 2;
			elif y_test[i] >= 1:
				y_test[i] = 1;
			else:
				y_test[i] = 0;
	#Collapse to 3 classifications => (0: Never used, 1: Used over a year ago, 2: Used within the year)
	elif _type == 3:
		for i in range(len(y_train)):
			if y_train[i] >= 3:
				y_train[i] = 2;
			elif y_train[i] >= 1:
				y_train[i] = 1;
			else:
				y_train[i] = 0;
		for i in range(len(y_test)):
			if y_test[i] >= 3:
				y_test[i] = 2;
			elif y_test[i] >= 1:
				y_test[i] = 1;
			else:
				y_test[i] = 0;
	#Keep original
	else:
		for i in range(len(y_train)):
			y_train[i] = int(y_train[i]);
		for i in range(len(y_test)):
			y_test[i] = int(y_test[i]);

def generate_predictions_filename(drug_name,reduction_name,algorithim_name,collapseType,cols):
    return (drug_name + "_" + reduction_name + "_" + algorithim_name + "_" + str(collapseType) + "_" + ','.join(str(i) for i in cols) + ".csv")

def get_reduction_name(reduction):
    if reduction.lower() == 'none' or reduction.lower() == 'no':
        reduction_name = 'None'
    elif reduction.lower() == 'pca':
        reduction_name = 'PCA'
    elif reduction.lower() == 'fld':
        reduction_name = 'FLD'
    else:
        eprint("\nERROR: Reduction method was not found\n");
        sys.exit();
    return reduction_name

def get_algorithim_name(algorithim):
    if algorithim.lower() == 'clustering' or algorithim.lower() == 'cluster':
        algorithim_name = 'Clustering'
    elif algorithim.lower() == 'decisiontree' or algorithim.lower() == 'dt':
        algorithim_name = 'DecisionTree'
    elif algorithim.lower() == 'knn':
        algorithim_name = 'kNN'
    elif algorithim.lower() == 'bpnn':
        algorithim_name = 'BPNN'
    elif algorithim.lower()  == 'mpp':
        algorithim_name = 'MPP'
    else:
        eprint("\nERROR: Algorithim was not found\n");
        sys.exit();
    return algorithim_name

def main():
    printHelp();

    if len(sys.argv) < 6:
        eprint("\nERROR: Not enough arguments given to the program! Please refer to the above help section.\n");
        sys.exit();

    reduction_name = get_reduction_name(sys.argv[1]);
    algorithim_name = get_algorithim_name(sys.argv[2]);
    filename = sys.argv[3]

    drug_name = os.path.splitext(os.path.basename(filename))[0]

    collapseType = int(sys.argv[4]);

    if len(sys.argv) > 6 and sys.argv[6][0].lower() == 't':
        random.seed(1)
        save_predictions = True
    else:
        save_predictions = False

    if not ((0 <= collapseType <= 3)):
        eprint("\nERROR: Collapse type must be between 0-3\n")
        sys.exit()

    cols = sys.argv[5].split(',');
    for i in range(len(cols)):
        cols[i] = int(cols[i]);
        if not (-1 <= cols[i] <= 11):
            eprint("\nERROR: Collumns must be between -1-11\n")
            sys.exit()

    #Sort for saving filename purpose
    cols = sorted(cols)

    split = .67;
    X_train = [];
    y_train = [];
    X_test = [];
    y_test = [];
    loadDataset(filename,split,cols,X_train,y_train,X_test,y_test);
    collapseClassifications(collapseType,y_train,y_test);

    if save_predictions == True:
        random.seed(datetime.now())
        predictions_filename = generate_predictions_filename(drug_name,reduction_name,algorithim_name,collapseType,cols)
    else:
        predictions_filename = None

    print("\nRunning algorithim... ");
    print("\tReduction     : " + str(reduction_name))
    print("\tAlgorithim    : " + str(algorithim_name))
    print("\tCollapse Type : " + str(collapseType))
    print("\tColumns Used  : " + str(cols))


    if reduction_name.lower() == 'pca':
        X_train,X_test = PCA.run(X_train,X_test)
    elif reduction_name.lower() == 'fld':
        X_train,X_test = FLD.run(X_train,y_train,X_test)

    if algorithim_name.lower() == 'clustering':
        accuracy,classifier,confusion_matrix,precision,recall,f1 = clustering.run(X_train,y_train,X_test,y_test,predciction_filename=predictions_filename);
    elif algorithim_name.lower() == 'decisiontree':
        save_decision_tree = False
        if save_decision_tree == True and reduction_name == 'None' and len(cols) == 1 and cols[0] == -1:
            save_decision_tree = True
        else:
            save_decision_tree = False
        accuracy,classifier,confusion_matrix,precision,recall,f1 = DecisionTree.run(X_train,y_train,X_test,y_test,outputGraph=save_decision_tree,collapseType=collapseType,predciction_filename=predictions_filename);
    elif algorithim_name.lower() == 'knn':
        accuracy,classifier,confusion_matrix,precision,recall,f1 = kNN.run(X_train,y_train,X_test,y_test,predciction_filename=predictions_filename);
    elif algorithim_name.lower() == 'bpnn':
        accuracy,classifier,confusion_matrix,precision,recall,f1 = BPNN.run(X_train,y_train,X_test,y_test,predciction_filename=predictions_filename);
    elif algorithim_name.lower()  == 'mpp':
        accuracy,classifier,confusion_matrix,precision,recall,f1 = MPP.run(X_train,y_train,X_test,y_test,predciction_filename=predictions_filename);

    # Not an error but allows run_experiments.sh to see this output
    eprint('\nAccuracy         : {}'.format(accuracy))
    eprint('Best Parameters  :  {}\n'.format(classifier))
    print('Confusion Matrix : \n{}\n'.format(confusion_matrix))

	#Append results
    directory = "Results";
    if not os.path.exists(directory):
        os.makedirs(directory)

    string = "";
    with open(os.path.join(directory,algorithim_name+'.csv'), 'a') as file:
        string = drug_name;
        string = string + ',' + (str(reduction_name)).replace(',','');
        string = string + ',' + (str(collapseType)).replace(',','');
        string = string + ',' + (str(cols)).replace(',','');
        string = string + ',' + (str(accuracy)).replace(',','');
        string = string + ',' + (str(classifier)).replace(',','');
        string = string + ',' + (str(confusion_matrix.tolist())).replace(',','');
        string = string + ',' + (str(precision)).replace(',','');
        string = string + ',' + (str(recall)).replace(',','');
        string = string + ',' + (str(f1)).replace(',','');

        file.write(string+"\n");

if __name__ == "__main__":
	main()
