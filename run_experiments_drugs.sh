#!/bin/bash
# CONTROL + Z will terminate script
# Run below command to make script executable
# chmod +x run_experiments.sh

# terminate after an error
# set -e

for filename in 'DrugData/IllegalDrugs.csv' 'DrugData/LegalDrugs.csv'
do
   for reduction_method in 'None' 'FLD' 'PCA'
   do
      for algorithim_name in 'Clustering' 'DecisionTree' 'kNN' 'BPNN' 'MPP'
      do
         for collapse_type in 0 1 2 3
         do
            for columns_to_use in 0,1,2,3,4 5,6,7,8,9,10,11 -1
            do
               echo "<<--- EXPERIMENT --->> "
               echo " -- Filename       : " $filename
               echo " -- Reduction      : " $reduction_method
               echo " -- Algorithim     : " $algorithim_name
               echo " -- Collapse Type  : " $collapse_type
               echo " -- Columns to Use : " $columns_to_use

               python python/runner.py $reduction_method $algorithim_name $filename $collapse_type $columns_to_use > /dev/null
            done
         done
      done
   done
done
