# Final-Project-ECE471:

The goal of this project is to use various machine learning algorithms to classify the below dataset. 10 fold cross validation is used to determine the best parameters and results are gathered on unseen data.

# Links:
Dataset:<br>
http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29 <br>

Results.csv:<br>
https://docs.google.com/spreadsheets/d/1E3bBgM-ivO9hUzD8B0ikVWiMWTvuq05GDoG59lepFuY/edit?usp=sharing <br>

Report:<br>
https://docs.google.com/document/d/1GISlAoKiMtV66OsgX6d8238h52S9VA8Tq2iWwKQgYVI/edit?usp=sharing <br>

Powerpoint:<br>
https://docs.google.com/presentation/d/1lzonRb6BPj5XVkpVQKUBtOgfzcfdzW76sOFiuFKhsEc/edit?usp=sharing

# Datasets:
Data_Drug_Consumption/sub_data/:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;All data used in experiments. Each file contains a particular drug, its features and classification. This was created by stripping it from the original dataset.

Features:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Personality measurements which include NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), level of education, age, gender, country of residence and ethnicity.

Class Labels:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"Never Used", "Used over a Decade Ago", "Used in Last Decade", "Used in Last Year", "Used in Last Month", "Used in Last Week", and "Used in Last Day".

# Runner.py:
Params: REDUCTION_NAME ALGORITHIM_NAME DATASET_FILENAME COLLAPSE_TYPE COLUMNS_TO_USE (Optional) VERBOSE <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;REDUCTION_NAME: PCA, FLD, None <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ALGORITHIM_NAME: BPNN, clustering (WTA and kMeans), DecisionTree, kNN, MPP (case1,2,3), SVM, Random forest classifier, AdaBoostClassifier <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DATASET_FILENAME: Any file in ... "Data_Drug_Consumption/sub_data" <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COLLAPSE_TYPE: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: None, use original classes <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: Collapse to 2 classes. (0: Never used, 1: Used at somepoint)  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2: Collapse to 3 classes. (0: Never used, 1: Used over a decade ago, 2: Used within the decade)  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3: Collapse to 3 classes. (0: Never used, 1: Used over a year ago, 2: Used within the year)  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;COLUMNS_TO_USE: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0-11 -1 for all
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0-4  : Personal Info (Age, Gender, Education, Country, Ethnicity)  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5-11 : Personality Traits (Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, sensation)  <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-1   : All <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VERBOSE: True or False <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True: Save predictions, plot as many graphs as applicable to config

Description:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Will run the specified configuration on the dataset using algorithm name    reduction type, collapseType, and columns to use. It will then print the results and append them to the "Results" directory. Optionally, true as the last parameter will save predictions to "Predictions" directory and show as many graphs as possible.

# fusion.py:
Params: first_predictions.csv second_predictions.csv <br>

Description: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fuse two classifiers using Naive bayesian fusion and print the results. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Both predictions files are generated by running runner.py with the optional final parameter set to true. It will be stored in the "Predictions" directory.


# run_experiments.sh:
Description:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Chmod before using. Will conduct a large list of experiments and save their results in a directory called "Results"

Current Warnings:
1) There will potentially be warnings for the performance metrics if a confusion matrix has a zero value for any cell. This warning can be ignored.
