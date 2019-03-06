# -*- coding: utf-8 -*-
import warnings
from collections import OrderedDict
import numpy as np
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

#Sampling
from sklearn.model_selection import train_test_split

# Classifiier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

#Metrics
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

titanic_clean_df = pd.read_csv('C:\\dev\Demos\\BAS2018\\Titanic_Python\\final_titanic_clean.csv')

target_df = titanic_clean_df['survived']
features_df = titanic_clean_df.drop(['survived', 'died'],axis=1)
                                 
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size = 0.3, random_state=42)

# Logistic Regression : 
lr = LogisticRegression()

# Fit the regressor to the training data
lr.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = lr.predict(X_test)

# Score / Metrics
lr_accuracy = lr.score(X_test, y_test)
prob = lr.predict_proba(X_test)[:,1]

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, prob)

precision, recall, thresholds = precision_recall_curve(y_test, prob)

roc_auc = auc(fpr, tpr)

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

##############################################################################
# Create a random Forest Classifier instance
rfc = RandomForestClassifier(n_estimators = 100)
# Fit to the training data
rfc.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = rfc.predict(X_test)

# Score / Metrics
rfc_accuracy = rfc.score(X_test, y_test) # = accuracy

##############################################################################
#Decision Tree
dtree = DecisionTreeClassifier(max_depth=10)

dtree.fit(X_train,  y_train)
y_pred = dtree.predict(X_test)

dtree_accuracy = dtree.score(X_test,  y_test) # = accuracy
    
#Features importance
FeaturesImportance(features_df,rfc)
FeaturesImportance(features_df,dtree)

#Visualise
with open("C:\\dev\Demos\\BAS2018\\Titanic_Python\\titanic_tree.txt", "w") as f:
    f = export_graphviz(dtree, out_file=f)

print("Logistic Regression: " + str(lr_accuracy))
print("Random Forest: " + str(rfc_accuracy))
print("Decison Tree: " + str(dtree_accuracy))