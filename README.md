# Credit_Risk_Analysis

OVERVIEW OF THE PROJECT:


Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, 
you’ll oversample the data using the RandomOverSampler and SMOTE algorithms, 
and undersample the data using the ClusterCentroids algorithm. 
Then, you’ll use a combinatorial approach of over and undersampling using the SMOTEENN algorithm. 
Next, you’ll compare two new machine learning models that reduce bias,
 BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.
 Once you’re done, you’ll evaluate the performance of these models and make a written recommendation 
on whether they should be used to predict credit risk.

DELIVERABLES:
This new assignment consists of three technical analysis deliverables and a written report.

Deliverable 1: Use Resampling Models to Predict Credit Risk
Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
Deliverable 4: A Written Report on the Credit Risk Analysis README.md
Deliverables:
This new assignment consists of three technical analysis deliverables and a proposal for further statistical study:

Data Source: Module-17-Challenge-Resources.zip and LoanStats_2019Q1.csv
Data Tools: credit_risk_resampling_starter_code.ipynb and credit_risk_ensemble_starter_code.ipynb.
Software: Python 3.9, Visual Studio Code 1.50.0, Anaconda 4.8.5, Jupyter Notebook 6.1.4 and Pandas

DELIVERABLE 1-------

Using your knowledge of the imbalanced-learn and scikit-learn libraries,  evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First,  useD the oversampling RandomOverSampler and SMOTE algorithms, and then you’ll use the undersampling ClusterCentroids algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

To Deliver.

Follow the instructions below:

Follow the instructions below and use the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 1.

Open the credit_risk_resampling_starter_code.ipynb file, rename it credit_risk_resampling.ipynb, and save it to your Credit_Risk_Analysis folder.

Using the information we’ve provided in the starter code, create your training and target variables by completing the following steps:

Create the training variables by converting the string values into numerical ones using the get_dummies() method.
Create the target variables.
Check the balance of the target variables.
Next, begin resampling the training data. First, use the oversampling RandomOverSampler and SMOTE algorithms to resample the data, then use the undersampling ClusterCentroids algorithm to resample the data. For each resampling algorithm, do the following:

Use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
Calculate the accuracy score of the model.
Generate a confusion matrix.
Print out the imbalanced classification report.
Save your credit_risk_resampling.ipynb file to your Credit_Risk_Analysis folder.

Deliverable 1 Requirements

For all three algorithms, the following have been completed:

An accuracy score for the model is calculated
A confusion matrix has been generated
An imbalanced classification report has been generated.

DELIVERABLE  2:--------------

Use the SMOTEENN algorithm to Predict Credit Risk.
We use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 2.

Continue using your credit_risk_resampling.ipynb file where you have already created your training and target variables.
Using the information we have provided in the starter code, resample the training data using the SMOTEENN algorithm.
After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Save your credit_risk_resampling.ipynb file to your Credit_Risk_Analysis folder.

Deliverable 2 Requirements
The combinatorial SMOTEENN algorithm does the following:

An accuracy score for the model is calculated
A confusion matrix has been generated
An imbalanced classification report has been generated

DELIVERABLE 3:---------------

Use Ensemble Classifiers to Predict Credit Risk
Deliverable Requirements:
Using your knowledge of the imblearn.ensemble library, you’ll train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

To Deliver.

Follow the instructions below:

Follow the instructions below and use the information in the credit_risk_resampling_starter_code.ipynb file to complete Deliverable 3.

Open the credit_risk_ensemble_starter_code.ipynb file, rename it credit_risk_ensemble.ipynb, and save it to your Credit_Risk_Analysis folder.
Using the information we have provided in the starter code, create your training and target variables by completing the following:
Create the training variables by converting the string values into numerical ones using the get_dummies() method.
Create the target variables.
Check the balance of the target variables.
Resample the training data using the BalancedRandomForestClassifier algorithm with 100 estimators.
Consult the following Random Forest documentation for an example.
After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Print the feature importance sorted in descending order (from most to least important feature), along with the feature score.
Next, resample the training data using the EasyEnsembleClassifier algorithm with 100 estimators.
Consult the following Easy Ensemble documentation for an example.
After the data is resampled, use the LogisticRegression classifier to make predictions and evaluate the model’s performance.
Calculate the accuracy score of the model, generate a confusion matrix, and then print out the imbalanced classification report.
Save your credit_risk_ensemble.ipynb file to your Credit_Risk_Analysis folder.

Deliverable 3 Requirements:

The BalancedRandomForestClassifier algorithm does the following:

An accuracy score for the model is calculated
A confusion matrix has been generated
An imbalanced classification report has been generated
The features are sorted in descending order by feature importance
The EasyEnsembleClassifier algorithm does the following:

An accuracy score of the model is calculated
A confusion matrix has been generated
An imbalanced classification report has been generated.


**************DELIVERABLE RESULTS:***************

Below are the results from the various techniques used to predictive model for High-Risk loans.

The results for the six machine learning models including their respective balanced accuracy, precision, and recall scores are as follows:

***Naive Random Oversampling


Balanced Accuracy: 0.6438627638488825

Precision: The precision is HIGH for High-risk loans and is LOW for Low-risk loans.

Recall: High/Low risk = .69/.60

***SMOTE Oversampling


Balanced Accuracy: 0.6628910844779521

Precision: The precision is LOW for High-risk loans and is HIGH for Low-risk loans.

Recall: High/Low risk = .63/.69

*****Undersampling


Balanced Accuracy: 0.6628910844779521

Precision: The precision is HIGH for High-risk loans and is LOW for Low-risk loans.

Recall: High/Low risk = .69/.40

*****Combination Under-Over Sampling

Balanced Accuracy: 0.5442661782548694

Precision: The precision is HIGH for High-risk loans and is LOW for Low-risk loans.

Recall: High/Low risk = .72/.57

******Balanced Random Forest Classifier


Balanced Accuracy: 0.8127490320735181

Precision: The precision is LOW for High-risk loans and is HIGH for Low-risk loans.

Recall: High/Low risk = .70/.92

*****Easy Ensemble AdaBoost Classifier


Balanced Accuracy: 0.9251352679776481

Precision: The precision is LOW for High-risk loans and is HIGH for Low-risk loans.

Recall: High/Low risk = 0.91/0.94

*********SUMMARY:*********************

For all models, utlizing EasyEnsembleClassifier is the most effective. Provides a highest Score for all Risk loans.
 The precision is low or none for all the models. In General, above the 90% of the current analysis,
 utlizing EasyEnsembleClassifier will perform a High-Risk loan precision as a great value for the overall analysis.