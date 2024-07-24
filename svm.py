import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # , roc_auc_score
# from sklearn.manifold import Isomap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib


def tsne_visual():
    tsne = TSNE(n_components=2, random_state=42)
    features_scaled = scaler.fit_transform(features)
    features_tsne = tsne.fit_transform(features_scaled)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Label')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


data = pd.read_csv("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
                   "\classifier_samples\\classifier_samples\\classifier_smot_multi_diagnosis2.csv")
features = data.drop(columns=['label'])
labels = data['label']
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)
# isomap = Isomap(n_components=5)
# features_train_isomap = isomap.fit_transform(features_train_scaled)
# features_test_isomap = isomap.transform(features_test_scaled)
rfe = RFE(estimator=GradientBoostingClassifier(random_state=42, n_estimators=100), n_features_to_select=30)
rfe.fit(features_train_scaled, labels_train)
features_train_rfe = rfe.transform(features_train_scaled)
features_test_rfe = rfe.transform(features_test_scaled)
svm = SVC(kernel='rbf', probability=True, random_state=42, C=10, gamma=0.1)

# param_grid = {
#     'C': [0.1, 1, 10, 50, 100],  # Regularization parameter
#     'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
#     'kernel': ['linear', 'rbf', 'poly']  # Kernel type
# }
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1,
#                            verbose=2)
# grid_search.fit(features_train_rfe, labels_train)
#
# # Best parameters from grid search
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')

svm.fit(features_train_rfe, labels_train)
pred = svm.predict(features_test_rfe)
prob = svm.predict_proba(features_test_rfe)[:, 1]
accuracy = accuracy_score(labels_test, pred)
# roc_auc = roc_auc_score(labels_test, pred)
print(f'Accuracy: {accuracy:.4f}')
# print(f'ROC-AUC: {roc_auc:.4f}')
print("Confusion Matrix:")
print(confusion_matrix(labels_test, pred))
print("\nClassification Report:")
print(classification_report(labels_test, pred))
cv_scores = cross_val_score(svm, features, labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
# tsne_visual()
# joblib_file = "svm_model3.pkl"
# joblib.dump(svm, joblib_file)
# print(f"Model saved to {joblib_file}")
