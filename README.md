# Credit_Risk_Analysis
Machine Learning/ Python/ Jupyter Notebook/ Data Preparation/ Statistical Reasoning

## Overview 
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, the purpose of of this analysis is to explore different techniques to train and evaluate machine learning models with unbalanced classes by utilizing imbalanced-learn and scikit-learn libraries. 

In the analysis, I will explore the following: 
* Oversampling the data using the RandomOverSampler and SMOTE algorithms
* Undersample the data using the ClusterCentroids algorithm
* Combinatorial approach of over and undersampling using the SMOTEENN algorithm
* Reduce bias via BalancedRandomForestClassifier and EasyEnsembleClassifier

## Results 
Random Oversampling
* In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. 
* In this study, the balanced accuracy score is ~65%, while precision and recall are at 0.99 and 0.69. 

Synthetic Minority Oversampling Technique (SMOTE)
* In SMOTE, like random oversampling, the size of the minority is increased. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated.
* The metrics in this instance have not changed much. The balanced accuracy score stays consistent at ~65%, while precision and recall are also consistent at 0.99 and 0.69. 

Random Undersampling
* Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.
* Generally, the results here are unimpressive, especially for predicting defaults. The balanced accuracy score comes in lower than previous methods at 55%, while precision and recall are at 0.99 and 0.40. 

Combination (Over and Under) Sampling (SMOTEENN)
* SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process - Oversample the minority class with SMOTE; 
Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
* Resampling with SMOTEENN did not work miracles, but some of the metrics show an improvement over undersampling. The balanced accuracy score went back up to ~65%, while the precision and recall are at 0.99 and 0.57. 

Ensemble Learners 
* Balanced Random Forest Classifier - Utiziling Ensemble Learners made a slight overall improvement in results. We see a balanced accuracy score of 79%, while precision and recall are at 0.99 and 0.87. 
* AdaBoost Classifer - Using this methond proved the best results, with a balanced accuracy score of 93%, while precision and recall are at 0.99 and 0.94

## Summary 
There are many ways to utilize machine learning to make educated predictions. In this analysis, 6 different methods were implemented and tested, each with its own set of pros and cons. Overall, the numbers speak volumes and give us predictive insights before making important decisions. In this case, it is clear that the AdaBoost Classifier method using Ensemble Learners reduced bias to predict credit risk the most effectively. 
