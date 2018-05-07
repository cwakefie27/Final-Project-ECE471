# Final-Project-ECE471

My idea for all the python programs is to use gridsearchCV to make finding the optimal parameters easier for us. To test parameters we will just have to put them in the params_grid and it will test all of them and return the best choice. Furthermore, it does this all in parallel and using 10 fold cross validation.

SEMER
- 1876 Never used (0)
- 2 Used over a Decade Ago (1)
- 3 Used in Last Decade (2)
- 2 Used in Last Year (3)
- 1 Used in Last Month (4)
- 0 Used in Last Week (5)
- 0 Used in Last Day (6)

Current Issues:

1) Maybe we should change to using Random search for parameters instead of GridSearchCV, SKLearn has this and it is an easy change
2) When should I terminate WTA? Currently, I am just terminating at the max iterations specified.
3) How to do classifier fusion efficiently
4) Does MPP have parameters, also there is a chance that a singular matrix occurs during cross validation, if this happens then that particular configuration cannot be predicted so the accuracy score will be zero. Is this correct?
5) To graph or not graph decision tree is hardcoded
6) There will potentially be warnings for the performance metrics, ignore them, they do not matter, they come from classes not being predicted

TODO:

1) FLD, Classifier Fusion
