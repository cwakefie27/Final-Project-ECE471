# Final-Project-ECE471

My idea for all the python programs is to use gridsearchCV to make finding the optimal parameters easier for us. To test parameters we will just have to put them in the params_grid and it will test all of them and return the best choice. Furthermore, it does this all in parallel and using 10 fold cross validation.


Current Issues:

1) Maybe we should change to using Random search for parameters instead of GridSearchCV, SKLearn has this and it is an easy change
2) When should I terminate WTA? Currently, I am just terminating at the max iterations specified.
3) How to do classifier fusion efficiently
4) Does MPP have parameters, also there is a chance that a singular matrix occurs during cross validation, if this happens then that particular configuration cannot be predicted so the accuracy score will be zero. Is this correct?
5) To graph or not graph decision tree is hardcoded

TODO:

1) FLD, Classifier Fusion
