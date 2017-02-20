# Rusan Machine Learning Final
Zeid M. Rusan  
2/19/2017  



##Summary
This report concerns the ability to predict exercise "correctness", or how well a routine was performed, based on on-body sensors. As suggested by Velloso et al. 2013, a Random Forest approach was used as a model because of "characteristic noise present in the sensor data". Non-NA sensor training data with 10-fold cross validation was used to train the random forest model, which was then used to predict exercise performance in test cases. The training set accuracy was **~99.53%** for the average model from 10 cross validations, which greatly minimized the out-of-sample error measured with the test data. Overall, the model was a powerful classifier of exericse accuracy when given new data.

##Methods
* **Data loading and feature selection:** 

  First data is read in, then columns with NA ("NA" or blank values) are removed. Next, time stamp and window data are excluded since they should not have anything to do with the "classe" variable type.

```r
  training <- read.csv("pml-training.csv",na.strings = c("","NA"))
  nonNAs <- which(apply(training,2,function(x) sum(is.na(x)))==0)
  training <- training[,nonNAs]
  testing <- read.csv("pml-testing.csv",na.strings = c("","NA"))
  testing <- testing[,nonNAs]
  variablesToKeep <- c(2,8:60) #User name and non-time stamp or window data + classe variable
  training <- training[,variablesToKeep]
  trainingX <- training[,-54]
  trainingY <- training[,54]
  testing <- testing[,variablesToKeep]
  testingX <- testing[,-54]
  testingY <- testing[,54]
```
* **Enable parallel processing using multiple cores for cross validation with caret:** 

This modification greatly increases machine learning efficiency and allowed for more cross validation within a reasonable time period.

```r
  library(caret)
  library(parallel)
  library(doParallel)
  cluster <- makeCluster(detectCores() - 1) # leave one free corse for OS
  registerDoParallel(cluster)
  fitControl <- trainControl(method = "cv",number = 10,allowParallel = TRUE)
```

* **Build model using cross validation with training set:**

A random forest with 10-fold cross validation was used to optimize the model. Cross validation training subsets were of size ~17659, and cross validation test subsets were of size ~ 1963. The model was used downstream for predictions using test cases.

```r
  fit <- train(trainingX,trainingY, method="rf",data=training,trControl = fitControl)
  stopCluster(cluster)
  registerDoSEQ()
```

* **The optimal model was used for prediction using test cases:**

```r
  prediction <- predict(fit,newdata = testing)
```

##Results
* **A random forest model using 27 variables accurately predicts training data:**

The model accuracy (average of resamples) was **~99.53%** using the training data.

```r
  fit
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17659, 17660, 17660, 17659, 17660, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9953627  0.9941340
##   27    0.9953116  0.9940695
##   53    0.9904191  0.9878795
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
  fit$resample
```

```
##     Accuracy     Kappa Resample
## 1  0.9943935 0.9929071   Fold02
## 2  0.9923586 0.9903342   Fold01
## 3  0.9949032 0.9935520   Fold03
## 4  0.9938838 0.9922633   Fold06
## 5  0.9969419 0.9961318   Fold05
## 6  0.9938869 0.9922673   Fold04
## 7  0.9959225 0.9948426   Fold07
## 8  0.9954152 0.9942007   Fold10
## 9  0.9974516 0.9967763   Fold09
## 10 0.9984702 0.9980648   Fold08
```

```r
  confusionMatrix.train(fit)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.3  0.1  0.0  0.0
##          C  0.0  0.0 17.3  0.2  0.0
##          D  0.0  0.0  0.0 16.1  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9954
```

* **The model generalizes well to the 20 test cases:**

Because we used cross-validation with random forests, I expect the out-of-sample error rate to roughly match those of the training set (~ 100 - 99.53 = 0.47%). Thus, the model is sufficient for prediction in new data test cases. Indeed, the model predicted 20/20 quick classifications correctly.

```r
  prediction
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
