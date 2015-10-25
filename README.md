R script for the Kaggle Titanic competition (https://www.kaggle.com/c/titanic/)

The script is my implementation of an approach described by Trevor Stephens 
(http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r)

Data are pre-processed by extracting passengers' titles (e.g., Mr.) from their names,
and estimating the number of immediate family members they are traveling with.  
Missing age values are imputed with a decision tree.  A few other missing values are 
imputed using median or naive likelihood-based assumptions.

Random forest and conditional random forest models are generated using different numbers of trees
(100, 500, 1000 or 2000) and written to .csv files for submission to the competition.

Code for an optional validation step (to compare models or perform a sanity check) is provided 
at the bottom of the script and is commented out.
