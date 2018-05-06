# Final-Project-ECE471

My idea for all the python programs is to use gridsearchCV to make finding the optimal parameters easier for us. To test parameters we will just have to put them in the params_grid and it will test all of them and return the best choice. Furthermore, it does this all in parallel and using 10 fold cross validation.

The next thing to do is figure out how we want classes sectioned (Does illegal drugs vs does not). Then make a python program for parsing those sections into variables X_train and y_train to those specs.

Feel free to change my implementations if you see something wrong and do not hesitate to push to the git.

Current Issues:

1) When should I terminate WTA? Currently, I am just terminating at the max iterations specified.
2) How to do classifier fusion efficiently
3) Does MPP have parameters, also there is a chance that a singular matrix occurs during cross validation, if this happens then that particular configuration cannot be predicted so the accuracy score will be zero

TODO:

1) FLD, Classifier Fusion
2) Decide how to split classes, create file parser
3) Combine everything
4) Blow brains out
