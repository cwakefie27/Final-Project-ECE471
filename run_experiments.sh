#!/bin/bash
# CONTROL + Z will terminate script
# Run below command to make script executable
# chmod +x run_experiments.sh

# terminate after an error
# set -e

# for drug_name in 'alcohol.csv' 'amphet.csv' 'amyl.csv' 'benzos.csv' 'caffeine.csv' 'cannabis.csv' 'chocolate.csv'
for filename in ./Data_Drug_Consumption/sub_data/*.csv
do
   for algorithim_name in 'Clustering' 'DecisionTree' 'kNN' 'BPNN' 'MPP'
   do
      for collapse_type in 0 1 2 3
      do
         for columns_to_use in -1
         do
            echo "<<--- EXPERIMENT --->> "
            echo " -- Filename       : " $filename
            echo " -- Algorithim     : " $algorithim_name
            echo " -- Collapse Type  : " $collapse_type
            echo " -- Columns to Use : " $columns_to_use
            # python python/runner.py $algorithim_name Data_Drug_Consumption/sub_data/$filename $collapse_type $columns_to_use > /dev/null
            python python/runner.py $algorithim_name $filename $collapse_type $columns_to_use > /dev/null
         done
      done
   done
done
