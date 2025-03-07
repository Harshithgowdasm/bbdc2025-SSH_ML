import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

class FraudDetection:
    def __init__(self, df, df_val):
        """
        Initialize the class with the dataset.
        """
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df_val = df_val
        self.x_val = None
        self.y_val = None
        self.scaler = StandardScaler()  # Initialize scaler

    def preprocess_data(self):
        """
        Preprocess the dataset: Separate features and target, scale features, and split into train/test sets.
        """
        # Separate features (X) and target (y)
        X = self.df.drop(columns=['Fraudster', 'AccountID'])  # Drop non-feature columns
        y = self.df['Fraudster']

        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        # Preprocess validation data
        self.x_val = self.df_val.drop(columns=['Fraudster', 'AccountID'])
        self.y_val = self.df_val['Fraudster']
        self.x_val = pd.DataFrame(self.scaler.transform(self.x_val), columns=self.x_val.columns)  # Use the same scaler
        self.X_train = X_scaled
        self.y_train = y
        self.y_test = self.y_val
        self.X_test= self.scaler.fit_transform(self.x_val)

        # Split the data into training and testing sets
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def oversample_data(self):
        """
        Apply SMOTE to oversample the minority class.
        """
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        return X_train_smote, y_train_smote

    def undersample_data(self):
        """
        Apply Random Under-Sampling to reduce the majority class.
        """
        under = RandomUnderSampler(random_state=42)
        X_train_under, y_train_under = under.fit_resample(self.X_train, self.y_train)
        return X_train_under, y_train_under

    def hybrid_sample_data(self):
        """
        Combine SMOTE and Random Under-Sampling for hybrid sampling.
        """
        smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Oversample minority class
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)  # Undersample majority class
        pipeline = Pipeline(steps=[('over', smote), ('under', under)])
        X_train_hybrid, y_train_hybrid = pipeline.fit_resample(self.X_train, self.y_train)
        return X_train_hybrid, y_train_hybrid

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, approach_name):
        """
        Train a model and evaluate its performance.
        """
        # Initialize the model
        clf_1 = GaussianNB()
        clf_2 = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
        clf_3 = MLPClassifier(hidden_layer_sizes=(300,), random_state=42)
        clf_4 = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', criterion='gini')
        clf_5 = LogisticRegression(random_state=42)
        eclf = VotingClassifier(estimators=[('gnb', clf_1), ('knn', clf_2), ('mlp', clf_3), ('rf', clf_4), ('lr', clf_5)], voting='hard')

        # Train the model
        model = eclf
        model.fit(X_train, y_train)

        # Make predictions on validation data
        y_pred = model.predict(X_test)

        # Evaluate the model
        print(f"Results for {approach_name}:")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def build_nn_model(self, input_shape):
        """
        Build a Neural Network model.
        """
        model = Sequential()
        model.add(Dense(64, input_dim=input_shape, activation='relu'))  # Input layer
        model.add(Dropout(0.5))  # Dropout for regularization
        model.add(Dense(32, activation='relu'))  # Hidden layer
        model.add(Dropout(0.5))  # Dropout for regularization
        model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def NN_train_and_evaluate_model(self, input_shape, X_train, y_train, X_test, y_test, approach_name, epochs=50, batch_size=32):
        """
        Train and evaluate a Neural Network model.
        """
        # Build the model
        model = self.build_nn_model(input_shape)

        # Train the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

        # Evaluate the model
        print(f"Results for {approach_name}:")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred):.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

        # Plot training history
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def compare_approaches(self):
        """
        Compare Oversampling, Undersampling, and Hybrid Approaches.
        """
        # Preprocess the data
        self.preprocess_data()

        # Initialize plot for ROC curves
        plt.figure(figsize=(8, 6))

        # Approach 1: Oversampling (SMOTE)
        X_train_smote, y_train_smote = self.oversample_data()
        input_shape = X_train_smote.shape[1]
        self.NN_train_and_evaluate_model(input_shape, X_train_smote, y_train_smote, self.X_test, self.y_test, "Oversampling (SMOTE)", epochs=50, batch_size=32)

        # Approach 2: Undersampling
        X_train_under, y_train_under = self.undersample_data()
        input_shape = X_train_under.shape[1]
        self.NN_train_and_evaluate_model(input_shape, X_train_under, y_train_under, self.X_test, self.y_test, "Undersampling", epochs=50, batch_size=32)

        # Approach 3: Hybrid Sampling
        X_train_hybrid, y_train_hybrid = self.hybrid_sample_data()
        input_shape = X_train_hybrid.shape[1]
        self.NN_train_and_evaluate_model(input_shape, X_train_hybrid, y_train_hybrid, self.X_test, self.y_test, "Hybrid Sampling", epochs=50, batch_size=32)

        # Finalize ROC Curve plot
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        plt.show()