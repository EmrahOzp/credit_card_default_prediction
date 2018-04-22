# credit_card_default_prediction
prediction of credit card defaults


ABSTRACT
This project aims to demonstrate the 7 predictive algorithms over customer default payments in Taiwan and compares the predictive accuracy
of probability of default among those methods. From the perspective of risk management, risk prediction is on the upstream 
for well- developed financial system where the major purpose of risk prediction is to use the financial information to predict business
performance or individual customer’s credit risk and to reduc the demage of uncertainity. However, from the perspective of risk control,
estimating the probability of default will be more meaningfull than classifying custimers into binary results ie. risks and non-risky. 
Therefore, the estimated real default probability is an important problem and an interesting challenge. 
     
INTRODUCTION
In the following context we review 7 predictive algorithms Logistic Regression, K-Nearest Neighbour, 
Support Vector Machine (SVM), Kernel – SVM, Naïve Bayes, Decision Tree Classification, 
Random forest Classification under 3 dimensionality reduction techniques PCA (Principal Component Analysis), 
LDA (Linear Discriminant Analysis) and Kernel PCA.  

DATA SET + BUSINESS PROBLEM DESCRIPTION
Default of Credit card Clients Dataset downloaded from UCI Machine Learning Repository Archive. 
Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
Characteristics: Multivariate, Classification, 30.000 entries and 24 attributes

Attribute information:
Dataset employs a binary variable as default payment (Yes = 1, No = 0), as dependent variable
23 explanatory variables, as independent variables:
Amount of given credit in US Dollar terms, Gender, Education, Maritial Status, Age, History of past payments
(payment status from April to September 2005), Measurement of the payment status (pay full, delay 1 month, delay 2 months, 
delay 3 months, delay 4 months, delay 5 months, delay 6 months, delay 7 months, delay 8 months, delay 9 months and above), 
Amount of previous payments (Amount paid in September to April) 


