import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class MLModels:
    def __init__(self, X, y, x_val, y_val):
        """
        Initialize the class with features (X) and target (y).
        :param X: Features (DataFrame or numpy array).
        :param y: Target (Series or numpy array).
        """
        self.X_train = X
        self.y_train = y
        self.X_test = x_val
        self.y_test = y_val
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate the model's performance using classification metrics.
        :param y_true: True labels.
        :param y_pred: Predicted labels.
        :param y_pred_proba: Predicted probabilities (for ROC-AUC).
        """
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print("\nClassification Report:\n", classification_report(y_true, y_pred))
        
        if y_pred_proba is not None:
            print("ROC-AUC Score:", roc_auc_score(y_true, y_pred_proba))
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            plt.plot(fpr, tpr, label='ROC Curve')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.show()

    def logistic_regression(self):
        """
        Train and evaluate a Logistic Regression model.
        """
        model = LogisticRegression(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def decision_tree(self):
        """
        Train and evaluate a Decision Tree model.
        """
        model = DecisionTreeClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def random_forest(self):
        """
        Train and evaluate a Random Forest model.
        """
        model = RandomForestClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def gradient_boosting(self):
        """
        Train and evaluate a Gradient Boosting model.
        """
        model = GradientBoostingClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def knn(self):
        """
        Train and evaluate a K-Nearest Neighbors (KNN) model.
        """
        model = KNeighborsClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def svm(self):
        """
        Train and evaluate a Support Vector Machine (SVM) model.
        """
        model = SVC(probability=True, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def naive_bayes(self):
        """
        Train and evaluate a Naive Bayes model.
        """
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def xgboost(self):
        """
        Train and evaluate an XGBoost model.
        """
        model = XGBClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)

    def lightgbm(self):
        """
        Train and evaluate a LightGBM model.
        """
        model = LGBMClassifier(random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        self.evaluate_model(self.y_test, y_pred, y_pred_proba)