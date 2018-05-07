from __future__ import print_function
import numpy as np
import csv
import os
import sys

#Parse dataset and convert any use of an illegal drug to a classifier.
#l is 1 for legal, 0 for illegal
def loadDataset(inputFile,outputFile,_l):
	illegalDrugs = [13,14,15,17,19,20,21,22,25,26,27,30];
	legalDrugs = [12,16,18,24,28];

	data = [];
	with open(inputFile,"rt") as csvfile:
		reader = csv.reader(csvfile);
		for row in reader:
			line = "";
			l = 0;
			i = 0;
			j = 0;
			for col in row:
				if j == 0:
					line = col;
				elif j <= 11:
					line = line + ','+col;
				else:
					for c in range(len(illegalDrugs)):
						if illegalDrugs[c] == j:
							if int(col) > i:
								i = int(col);
							break;
					for c in range(len(legalDrugs)):
						if legalDrugs[c] == j:
							if int(col) > l:
								l = int(col);
							break;
				j = j + 1;

			if _l == 1:
				line = line + ',' +str(l);
			else:
				line = line + ','+str(i);
			data.append(line);


	string = "";
	with open(os.path.join(outputFile), 'w') as file:
		for i in range(len(data)):
			file.write(data[i]+"\n");

def main():
	if len(sys.argv) < 4:
		eprint("\nERROR: Not enough arguments given to the program!\nPlease run by having Python parser.py INPUTFILE OUTPUTFILE LEGAL(0)_OR_ILLEGAL(1)");
		sys.exit();

	inputFile = sys.argv[1]
	outputFile = sys.argv[2]
	legal = int(sys.argv[3]);

	loadDataset(inputFile,outputFile,legal);

if __name__ == "__main__":
	main()
