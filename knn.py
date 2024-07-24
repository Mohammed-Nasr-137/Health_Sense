import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # , roc_auc_score
from sklearn.feature_selection import RFE
import joblib

data = pd.read_csv("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
                   "\classifier_samples\\classifier_samples\\classifier_smot_multi_diagnosis2.csv")
features = data.drop(columns=['label'])
labels = data['label']
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)
# rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=3)
rfe = RFE(estimator=GradientBoostingClassifier(random_state=42, n_estimators=100), n_features_to_select=28)
rfe.fit(features_train_scaled, labels_train)
features_train_rfe = rfe.transform(features_train_scaled)
features_test_rfe = rfe.transform(features_test_scaled)
# knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance')

# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to use
#     'weights': ['uniform', 'distance'],  # Weight function used in prediction
#     'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric to use
# }
# grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
#                            verbose=2)
# grid_search.fit(features_train_rfe, labels_train)
#
# # Best parameters from grid search
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')

knn.fit(features_train_rfe, labels_train)
pred = knn.predict(features_test_rfe)
accuracy = accuracy_score(labels_test, pred)
# roc_auc = roc_auc_score(labels_test, pred)
print(f'Accuracy: {accuracy:.4f}')
# print(f'ROC-AUC: {roc_auc:.4f}')
print("Confusion Matrix:")
print(confusion_matrix(labels_test, pred))
print("\nClassification Report:")
print(classification_report(labels_test, pred))
cv_scores = cross_val_score(knn, features, labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

joblib_file = "knn_model3.pkl"
joblib.dump(knn, joblib_file)
print(f"Model saved to {joblib_file}")
