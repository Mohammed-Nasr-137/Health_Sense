import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score  # , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report  # , roc_auc_score
# from sklearn.decomposition import PCA
# from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import joblib

warnings.filterwarnings(action='ignore', category=UserWarning, message=".*does not have valid feature names.*")


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


file_path = ("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
             "\classifier_samples\\classifier_samples\\classifier_smot_multi_diagnosis2.csv")
df = pd.read_csv(file_path)
features = df.drop(columns=['label'])
labels = df['label']
features_train, features_test, labels_train, labels_test = (
    train_test_split(features, labels, test_size=0.2, random_state=42))
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)
# pca = PCA(n_components=0.95) features_train_pca = pca.fit_transform(features_train_scaled) features_test_pca =
# pca.transform(features_test_scaled)
rfe = RFE(estimator=LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, C=10),
          n_features_to_select=29)
rfe.fit(features_train_scaled, labels_train)
features_train_rfe = rfe.transform(features_train_scaled)
features_test_rfe = rfe.transform(features_test_scaled)
# isomap = Isomap(n_components=20)
# features_train_isomap = isomap.fit_transform(features_train_scaled)
# features_test_isomap = isomap.transform(features_test_scaled)
# print("Number of PCA components: ", pca.n_components_)
log_reg = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000, C=10)

# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
#     'penalty': ['l2'],  # Only 'l2' penalty is supported by 'lbfgs' solver
#     'max_iter': [100, 300, 500, 700, 900, 1100],
#     'multi_class': ['multinomial', 'ovr']
# }
# grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
# grid_search.fit(features_train_scaled, labels_train)
#
# # Best parameters from grid search
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')

log_reg.fit(features_train_rfe, labels_train)
pred = log_reg.predict(features_test_rfe)
prob = log_reg.predict_proba(features_test_rfe)[:, 1]
print("Confusion Matrix:")
print(confusion_matrix(labels_test, pred))
print("\nClassification Report:")
print(classification_report(labels_test, pred))
# print("\nROC-AUC Score:")
# print(roc_auc_score(labels_test, pred))
cv_scores = cross_val_score(log_reg, features, labels, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
tsne_visual()

# joblib_file = "logistic_regression_model3.pkl"
# joblib.dump(log_reg, joblib_file)
#
# print(f"Model saved to {joblib_file}")
