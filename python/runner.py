from __future__ import print_function
import numpy as np
import random
import sys
import clustering
import DecisionTree
import kNN
import MPP
import BPNN
import MPP
import csv
import os
import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

#Print out information on how to run the program
def printHelp():
	print("\nPlease run the program with the following arguments: runner.py ALGORITHIM_NAME DATASET_FILENAME COLLAPSE_TYPE COLUMNS_TO_USE\n");
	print("Algorithims to Use: Clustering, DecisionTree, KNN, BPNN, MPP\n");
	print("Dataset Location: Data_Drug_Consumption/sub_data/\n");
	print("Classification Collapse Types: \n");
	print("	0: None, use original classes");
	print("	1: Collapse to 2 classes. (0: Never used, 1: Used at somepoint)");
	print("	2: Collapse to 3 classes. (0: Never used, 1: Used over a decade ago, 2: Used within the decade)");
	print("	3: Collapse to 3 classes. (0: Never used, 1: Used over a year ago, 2: Used within the year)");
	print("\n");
	print("Columns To Use: Please provide a comma seperated string of column indices to use. NO SPACES! ex: 0,3,4 to use the columns 0, 3, and 4.")
	print("	Put -1 to use all columns")

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

printHelp();

if len(sys.argv) < 5:
	print("\nNot enough arguments given to the program! Please refer to the above help section.\n");
	sys.exit();

print("\nRunning algorithim...");

algorithim = sys.argv[1];
filename = sys.argv[2];
collapseType = int(sys.argv[3]);

cols = sys.argv[4].split(',');
for i in range(len(cols)):
	cols[i] = int(cols[i]);

split = .67;
X_train = [];
y_train = [];
X_test = [];
y_test = [];
loadDataset(filename,split,cols,X_train,y_train,X_test,y_test);
collapseClassifications(collapseType,y_train,y_test);

if algorithim.lower() == 'clustering':
	algorithim_name = 'Clustering'
	accuracy,classifier = clustering.run(X_train,y_train,X_test,y_test);
elif algorithim.lower() == 'decisiontree' or algorithim.lower() == 'dt':
	algorithim_name = 'DecisionTree'
	accuracy,classifier = DecisionTree.run(X_train,y_train,X_test,y_test,False);
	# accuracy,classifier = DecisionTree.run(X_train,y_train,X_test,y_test,True);
elif algorithim.lower() == 'knn':
	algorithim_name = 'kNN'
	accuracy,classifier = kNN.run(X_train,y_train,X_test,y_test);
elif algorithim.lower() == 'bpnn':
	algorithim_name = 'BPNN'
	accuracy,classifier = BPNN.run(X_train,y_train,X_test,y_test);
elif algorithim.lower()  == 'mpp':
	algorithim_name = 'MPP'
	accuracy,classifier = MPP.run(X_train,y_train,X_test,y_test);
else:
	print("\nAlgorithim was not found\n");
	sys.exit();

eprint('\nAccuracy        : {}'.format(accuracy))
eprint('Best Parameters : {}'.format(classifier))
eprint("")

drug_name = os.path.splitext(os.path.basename(filename))[0]
directory = "Results";
if not os.path.exists(directory):
    os.makedirs(directory)

string = "";
with open(os.path.join(directory,algorithim_name+'.csv'), 'a') as file:
	string = drug_name;
	string = string + ',' + (str(collapseType)).replace(',','');
	string = string + ',' + (str(cols)).replace(',','');
	string = string + ',' + (str(accuracy)).replace(',','');
	string = string + ',' + (str(classifier)).replace(',','');

	file.write(string+"\n");
