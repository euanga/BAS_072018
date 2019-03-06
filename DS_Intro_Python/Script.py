import warnings
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

titanic_df = pd.read_csv('C:\\dev\Demos\\BAS2018\\DS_Intro_Python\\titanic_clean.csv')
target = titanic_df.survived
features = titanic_df.drop(['survived'],axis=1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state=42)
target = titanic_df.survived

# Logistic Regression : 
lr = LogisticRegression()
# Fit the regressor to the training data
lr.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = lr.predict(X_test)

# Score / Metrics
accuracy = lr.score(X_test, y_test)
prob = lr.predict_proba(X_test)[:,1]

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(y_test, prob)

precision, recall, thresholds = precision_recall_curve(y_test, prob)
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)

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

dt = DecisionTreeClassifier(max_depth=10)

accuracy = dt.fit(X_train,  y_train)
prob = dt.score(X_test,  y_test)

features = titanic_df.columns.tolist()
fi = dt.feature_importances_
sorted_features = {}
for feature, imp in zip(features, fi):
    sorted_features[feature] = round(imp,3)

# sort the dictionnary by value
sorted_features = OrderedDict(sorted(sorted_features.items(),reverse=True, key=lambda t: t[1]))

features_df = pd.DataFrame(list(sorted_features.items()), columns=['Features', 'Importance'])
plt.figure(figsize=(15, 5))
sns.barplot(x='Features', y='Importance', data=features_df);
plt.xticks(rotation=90) 
plt.show()

