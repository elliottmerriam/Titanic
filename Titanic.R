# VARIABLE DESCRIPTIONS (from https://www.kaggle.com/c/titanic-gettingStarted/data):
#   survival        Survival
# (0 = No; 1 = Yes)
# pclass          Passenger Class
# (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
# (C = Cherbourg; Q = Queenstown; S = Southampton)
# 
# SPECIAL NOTES:
#   Pclass is a proxy for socio-economic status (SES)
# 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower
# 
# Age is in Years; Fractional if Age less than One (1)
# If the Age is Estimated, it is in the form xx.5
# 
# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.
# 
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic
# 
# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.

############################################################################################

# This code is my implementation of the approach used by Trevor Stephens 
# (see http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r)
# Clear Workspace
rm(list=ls())
# Clear Console:
cat("\014")

# install.packages("randomForest")
# install.packages("party")

library(randomForest)
library(party)

# Import train and test data
train = read.csv("~/Downloads/train.csv") # set appropriate path for your computer
test = read.csv("~/Downloads/test.csv") # set appropriate path for your computer

## FEATURE ENGINEERING 

# Combine train and test data
test$Survived = NA
Data = rbind(train, test)

# Extract titles (e.g. Mr.) from names
Data$Name = as.character(Data$Name)
Data$Title = sapply(Data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
Data$Title = sub(' ', '', Data$Title)

# inspect titles
table(Data$Title)

#clean up titles for consistency
Data$Title[Data$Title %in% c('Mme', 'Mlle')] = 'Mlle'
Data$Title[Data$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
Data$Title[Data$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'

# make Title a factor
Data$Title = factor(Data$Title)

# Obtain size of passenger's on-board family from SibSp and Parch (+1 for self)
Data$FamilySize = Data$SibSp + Data$Parch + 1

# Parse out families by last name and attempt to avoid pooling
# non-related people who coincidentally share the same last name

# Isolate last name
Data$Surname = sapply(Data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
Data$FamilyID = paste(as.character(Data$FamilySize), Data$Surname, sep="")

# label all families of 3 or less as "small"
Data$FamilyID[Data$FamilySize <= 3] = 'Small'

# make FamilyID a factor
Data$FamilyID = factor(Data$FamilyID)

## IMPUTE MISSING VALUES 

# impute missing age values based on a decision tree
Agefit = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
               Data[!is.na(Data$Age),], method="anova")
Data$Age[is.na(Data$Age)] = predict(Agefit, Data[is.na(Data$Age),])

# impute missing ports of embarcation by assuming Southampton
Data$Embarked[which(Data$Embarked == '')] = "S"

# impute missing Fares with median fare
Data$Fare[is.na(Data$Fare)]=median(Data$Fare, na.rm=TRUE)

# train random forest models of different sizes and save the resulting predictions for submission
train = Data[1:891,]
test = Data[892:1309,]
set.seed(415)
N=100
setwd("~/Desktop/Kaggle/Projects/Titanic")
for (N in c(100,500,1000,2000)) {
  
  ## RANDOM FOREST MODEL
  rf = randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 
                       Title + FamilySize + FamilyID, data=train, importance=TRUE, ntree=N)
  # varImpPlot(fit)

  # generate predictions to submit for the test data
  Prediction = predict(rf, test)
  submit = data.frame(PassengerId = test$PassengerId, Survived = Prediction)
  write.csv(submit, file = paste("Titanic Engineered RF nTrees=",as.character(N),".csv"), row.names = FALSE)

  ## CONDITIONAL RANDOM FOREST MODEL
  crf = randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 
                      Title + FamilySize + FamilyID, data=train, importance=TRUE, ntree=N)
  
  # generate predictions to submit for the test data
  Prediction = predict(crf, test, OOB=TRUE, type = "response")
  submit = data.frame(PassengerId = test$PassengerId, Survived = Prediction)
  write.csv(submit, file = paste("Titanic Engineered CRF nTrees=",as.character(N),".csv"), row.names = FALSE)
  
}

######
## FOR VALIDATION PRIOR TO TEST PREDICTIONS
# train = Data[1:700,]
# valid = Data[701:891,]
# 
# set.seed(415)
# N=2000
# rf = randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 
#                     Title + FamilySize + FamilyID, data=train, importance=TRUE, ntree=N)
# varImpPlot(rf)
# Pred=predict(rf, valid)
# 
# crf = cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 
#                 Title + FamilySize + FamilyID, data = train, controls=cforest_unbiased(ntree=N, mtry=3))
# cPred = predict(crf, valid, OOB=TRUE, type = "response")
# 
# 
# Validation = data.frame(Survived=valid$Survived, Pred=Pred, cPred=cPred) 
# 
# Acc=sum(as.numeric(Validation$Survived==Validation$Pred))/nrow(Validation)
# cAcc=sum(as.numeric(Validation$Survived==Validation$cPred))/nrow(Validation)
# N
# Acc
# cAcc
