## MODEL BUILDING with Random Forest Classifier 

import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sn

# Load the cleaned dataset
dataset = pd.read_csv('new_churn_data.csv')

## Data Preparation (keeping the same preprocessing steps)
user_identifier = dataset['user']
dataset = dataset.drop(columns=['user'])

# One-hot encoding for categorical variables
dataset = pd.get_dummies(dataset)
dataset = dataset.drop(columns=['housing_na', 'zodiac_sign_na', 'payment_type_na'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    dataset.drop(columns='churn'), 
    dataset['churn'],
    test_size=0.2,
    random_state=0
)

# Handle class imbalance using undersampling
y_train.value_counts()
pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(0)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes]
y_train = y_train[new_indexes]

# Feature scaling (Random Forest doesn't strictly require it, but good practice)
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

## Random Forest Classifier (REPLACING Logistic Regression)
rf_classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    random_state=0,
    max_depth=10,           # Limit tree depth to prevent overfitting
    min_samples_split=5,
    min_samples_leaf=2
)

rf_classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = rf_classifier.predict(X_test)

# Evaluating Results
cm = confusion_matrix(y_test, y_pred)
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print("Precision: %0.4f" % precision_score(y_test, y_pred))
print("Recall: %0.4f" % recall_score(y_test, y_pred))
print("F1-Score: %0.4f" % f1_score(y_test, y_pred))

# Confusion Matrix Visualization
df_cm = pd.DataFrame(cm, index=(0, 1), columns=(0, 1))
plt.figure(figsize=(10, 7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(
    estimator=rf_classifier, 
    X=X_train, 
    y=y_train, 
    cv=10,
    scoring='accuracy'
)
print("RF Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

## Feature Importance Analysis (Random Forest advantage over Logistic Regression)
feature_importance = pd.DataFrame({
    'features': X_train.columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['features'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

## Recursive Feature Elimination with Random Forest
from sklearn.feature_selection import RFE

rfe = RFE(rf_classifier, n_features_to_select=20, step=10)
rfe = rfe.fit(X_train, y_train)

print("\nSelected Features by RFE:")
selected_features = X_train.columns[rfe.support_]
print(selected_features)

# Train final model with selected features
rf_classifier_rfe = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=10)
rf_classifier_rfe.fit(X_train[selected_features], y_train)

# Final predictions
y_pred_rfe = rf_classifier_rfe.predict(X_test[selected_features])

print("\nRFE Model Performance:")
print("Accuracy: %0.4f" % accuracy_score(y_test, y_pred_rfe))
print("F1-Score: %0.4f" % f1_score(y_test, y_pred_rfe))

# Cross-validation for RFE model
accuracies_rfe = cross_val_score(rf_classifier_rfe, X_train[selected_features], y_train, cv=10)
print("RF RFE Accuracy: %0.3f (+/- %0.3f)" % (accuracies_rfe.mean(), accuracies_rfe.std() * 2))

# Final Results Formatting
final_results = pd.concat([y_test, user_identifier], axis=1).dropna()
final_results['predicted_churn'] = y_pred_rfe  # Using RFE predictions
final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)
print("\nFinal Results Sample:")
print(final_results.head(10))

# Save results
final_results.to_csv('rf_churn_predictions.csv', index=False)
print("\nPredictions saved to 'rf_churn_predictions.csv'")
