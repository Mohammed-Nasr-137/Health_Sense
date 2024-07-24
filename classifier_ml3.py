import joblib
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, message=".*does not have valid feature names.*")


def init_clf(features_train, labels_train):
    log_reg = joblib.load("logistic_regression_model3.pkl")
    dt = joblib.load("decision_tree_model3.pkl")
    gb = joblib.load("gradient_boosting_model3.pkl")
    rf = joblib.load("random_forest_model3.pkl")
    knn = joblib.load("knn_model3.pkl")
    svm = joblib.load("svm_model3.pkl")

    voting_classifier = VotingClassifier(estimators=[
        ('log_reg', log_reg),
        ('dt', dt),
        ('gb', gb),
        ('rf', rf),
        ('knn', knn),
        ('svm', svm)
    ], voting='soft')
    voting_classifier.fit(features_train, labels_train)
    return voting_classifier


def pred(clf, sample):
    if isinstance(sample, pd.Series):
        sample = sample.values.reshape(1, -1)
    elif isinstance(sample, np.ndarray) and sample.ndim == 1:
        sample = sample.reshape(1, -1)
    predict = clf.predict(sample)
    return predict[0]


data = pd.read_csv("G:\\Uni\\1st\Second Term\Math\Linear Algebra\Project\Code\ptb-diagnostic-ecg-database-1.0.0"
                   "\classifier_samples\\classifier_samples\\classifier_smot_multi_diagnosis2.csv")
# print(data)
x = data.drop(columns=['label'])
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# classifier = joblib.load("voting_classifier3.pkl")
classifier = init_clf(x_train, y_train)
# cv_scores = cross_val_score(classifier, x, y, cv=5, scoring='accuracy')
# print("Cross-Validation Scores:", cv_scores)
# print("Mean CV Accuracy:", cv_scores.mean())
total = len(x_test)
success = 0
total_0 = 0
total_1 = 0
total_2 = 0
success_0 = 0
success_1 = 0
success_2 = 0
for i in range(total):
    # noinspection INSPECTION_NAME
    prediction = pred(classifier, x_test.iloc[i])
    if y_test.iloc[i] == 0:
        total_0 += 1
        if prediction == y_test.iloc[i]:
            success += 1
            success_0 += 1
    elif y_test.iloc[i] == 1:
        total_1 += 1
        if prediction == y_test.iloc[i]:
            success += 1
            success_1 += 1
    elif y_test.iloc[i] == 2:
        total_2 += 1
        if prediction == y_test.iloc[i]:
            success += 1
            success_2 += 1

print("total:", total, ", total 0:", total_0, ", total 1:", total_1, ", total 2:", total_2)
print("success:", success, ", success 0:", success_0, ", success 1:", success_1, ", success 2:", success_2)
print("accuracy:", success / total, ", accuracy 0:", success_0 / total_0, ", accuracy 1:", success_1 / total_1,
      ", accuracy 2:", success_2 / total_2)

# joblib_file = "voting_classifier2.pkl"
# joblib.dump(classifier, joblib_file)
# print(f"Model saved to {joblib_file}")
