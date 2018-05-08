# Final-Project-ECE471

ECE Final Project:

The purpose of this project is to use various machine learning algorithms to reduce data dimensions and classify the below dataset. 10 fold cross validation is used to determine the best parameters and results are gathered on unseen data.

Links:
   Dataset: http://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
   Results.csv: https://docs.google.com/spreadsheets/d/1E3bBgM-ivO9hUzD8B0ikVWiMWTvuq05GDoG59lepFuY/edit?usp=sharing
   Report:
   https://docs.google.com/document/d/1GISlAoKiMtV66OsgX6d8238h52S9VA8Tq2iWwKQgYVI/edit?usp=sharing
   Powerpoint:
   https://docs.google.com/presentation/d/1lzonRb6BPj5XVkpVQKUBtOgfzcfdzW76sOFiuFKhsEc/edit?usp=sharing

Data:
   Data_Drug_Consumption/sub_data/: All data used in experiments. Each file contains a particular drug, its features and classification. This was created by stripping it from the original dataset.

Features:
   Personality measurements which include NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), level of education, age, gender, country of residence and ethnicity.

Class Labels:
   "Never Used", "Used over a Decade Ago", "Used in Last Decade", "Used in Last Year", "Used in Last Month", "Used in Last Week", and "Used in Last Day".

Callable Scripts:
   runner.py:
      Params:
         REDUCTION_NAME ALGORITHIM_NAME DATASET_FILENAME COLLAPSE_TYPE COLUMNS_TO_USE
            REDUCTION_NAME: PCA, FLD, None
            ALGORITHIM_NAME: BPNN, clustering (WTA and kMeans), DecisionTree, kNN, MPP, SVM
            DATASET_FILENAME: Any file in ... "Data_Drug_Consumption/sub_data"
            COLLAPSE_TYPE:
               0: None, use original classes
               1: Collapse to 2 classes. (0: Never used, 1: Used at somepoint)
               2: Collapse to 3 classes. (0: Never used, 1: Used over a decade ago, 2: Used within the decade)
               3: Collapse to 3 classes. (0: Never used, 1: Used over a year ago, 2: Used within the year)
            COLUMNS_TO_USE: 0-11 -1 for all
               0-4  : Personal Info (Age, Gender, Education, Country, Ethnicity)
               5-11 : Personality Traids (Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, sensation)
               -1   : All

      Description:
         Will run the specified configuration on the dataset using algorithm name    reduction type, collapseType, and columns to use. It will then print the results and append them to the "Results" directory. Optionally, true as the last parameter will save predictions to "Predictions" directory and show as many graphs as possible.

   fusion.py:
      Params:

      Description:

   run_experiments.sh:
      Description:
         Chmod before using. Will conduct a large list of experiments and save their results in a directory called "Results"

Current Warnings:
   1) There will potentially be warnings for the performance metrics if a confusion matrix has a zero value for any cell. This warning can be ignored.
