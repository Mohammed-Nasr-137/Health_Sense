import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # , roc_auc_score
# from sklearn.manifold import Isomap
from sklearn.feature_selection import RFE
import joblib

data = pd.read_csv("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
                   "\classifier_samples\\classifier_samples\\classifier_smot_multi_diagnosis2.csv")
features = data.drop(columns=['label'])
labels = data['label']
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)
rfe_rf = RFE(
    estimator=RandomForestClassifier(random_state=42, n_estimators=500, min_samples_split=3, min_samples_leaf=31),
    n_features_to_select=25)
rfe_rf.fit(features_train, labels_train)
features_train_rfe_rf = rfe_rf.transform(features_train)
features_test_rfe_rf = rfe_rf.transform(features_test)
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=500, min_samples_split=3, min_samples_leaf=1)

rf_classifier.fit(features_train_rfe_rf, labels_train)
rfe_gb = RFE(estimator=GradientBoostingClassifier(random_state=42, n_estimators=100), n_features_to_select=18)
rfe_gb.fit(features_train, labels_train)
features_train_rfe_gb = rfe_gb.transform(features_train)
features_test_rfe_gb = rfe_gb.transform(features_test)
gb_classifier = GradientBoostingClassifier(random_state=42, n_estimators=100)

gb_classifier.fit(features_train_rfe_gb, labels_train)
rf_pred = rf_classifier.predict(features_test_rfe_rf)
rf_prob = rf_classifier.predict_proba(features_test_rfe_rf)[:, 1]
gb_pred = gb_classifier.predict(features_test_rfe_gb)
gb_prob = gb_classifier.predict_proba(features_test_rfe_gb)[:, 1]

print("rf metrics:")
accuracy = accuracy_score(labels_test, rf_pred)
# roc_auc = roc_auc_score(labels_test, rf_prob)
classification_rep = classification_report(labels_test, rf_pred)
print("Confusion Matrix:")
print(confusion_matrix(labels_test, rf_pred))
print(f"Accuracy: {accuracy}")
# print(f"ROC-AUC: {roc_auc}")
print("Classification Report:")
print(classification_rep)
cv_scores = cross_val_score(rf_classifier, features, labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


print("\ngb metrics:")
accuracy = accuracy_score(labels_test, gb_pred)
# roc_auc = roc_auc_score(labels_test, gb_prob)
classification_rep = classification_report(labels_test, gb_pred)
print("Confusion Matrix:")
print(confusion_matrix(labels_test, gb_pred))
print(f"Accuracy: {accuracy}")
# print(f"ROC-AUC: {roc_auc}")
print("Classification Report:")
print(classification_rep)
cv_scores = cross_val_score(gb_classifier, features, labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


print("\n dt metrics")
# isomap = Isomap(n_components=8)
# features_train_isomap = isomap.fit_transform(features_train)
# features_test_isomap = isomap.transform(features_test)
rfe_dt = RFE(estimator=DecisionTreeClassifier(random_state=42, criterion='log_loss', min_samples_leaf=2,
                                              min_samples_split=10), n_features_to_select=29)
rfe_dt.fit(features_train, labels_train)
features_train_rfe_dt = rfe_dt.transform(features_train)
features_test_rfe_dt = rfe_dt.transform(features_test)
dt_classifier = DecisionTreeClassifier(random_state=42, criterion='log_loss', min_samples_leaf=2, min_samples_split=10)

# param_grid = {
#     'criterion': ['gini', 'log_loss', 'entropy'],  # Regularization strength
#     'min_samples_split': [1, 3, 5, 7, 9, 11, 13],  # Only 'l2' penalty is supported by 'lbfgs' solver
#     'min_samples_leaf': [2, 4, 6, 8, 10, 12, 14]
# }
# grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
#                            verbose=2)
# grid_search.fit(features_train_rfe_dt, labels_train)
#
# # Best parameters from grid search
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')

dt_classifier.fit(features_train_rfe_dt, labels_train)
pred = dt_classifier.predict(features_test_rfe_dt)
prob = dt_classifier.predict_proba(features_test_rfe_dt)[:, 1]
accuracy = accuracy_score(labels_test, pred)
# roc_auc = roc_auc_score(labels_test, prob)
classification_rep = classification_report(labels_test, pred)
print("Confusion Matrix:")
print(confusion_matrix(labels_test, pred))
print(f"Accuracy: {accuracy}")
# print(f"ROC-AUC: {roc_auc}")
print("Classification Report:")
print(classification_rep)
cv_scores = cross_val_score(dt_classifier, features, labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

joblib_file_rf = "random_forest_model3.pkl"
joblib.dump(rf_classifier, joblib_file_rf)
print(f"Model saved to {joblib_file_rf}")
joblib_file_gb = "gradient_boosting_model3.pkl"
joblib.dump(gb_classifier, joblib_file_gb)
print(f"Model saved to {joblib_file_gb}")
joblib_file_dt = "decision_tree_model3.pkl"
joblib.dump(rf_classifier, joblib_file_dt)
print(f"Model saved to {joblib_file_dt}")
