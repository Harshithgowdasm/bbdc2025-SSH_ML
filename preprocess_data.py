import pandas as pd
from feature_extracter import *


def preprocess_and_extract_features():
    """
    Load datasets, preprocess, extract features, and handle missing values.
    Returns:
        df_fea (pd.DataFrame): Processed training data with extracted features.
        df_val_fea (pd.DataFrame): Processed validation data with extracted features.
        df_test_fea (pd.DataFrame): Processed test data with extracted features.
    """
    # Load your datasets
    df = pd.read_csv('task/train_set/x_train_aggregated.csv')  # Training aggregated data
    y_df = pd.read_csv('task/train_set/y_train.csv')  # Training labels
    df_val = pd.read_csv('task/validation_set/x_val_aggregated.csv')  # Validation aggregated data
    yval_df = pd.read_csv('task/validation_set/y_val.csv')  # Validation labels
    test_df = pd.read_csv('task/test_set/x_test.csv')  # Test data
    test_agg_df = pd.read_csv('task/test_set/x_test_aggregated.csv')  # Test aggregated data
    xtrain_df = pd.read_csv('task/train_set/x_train.csv')  # Training raw data
    x_val_df = pd.read_csv('task/validation_set/x_val.csv')  # Validation raw data

    # Merge training data with labels
    # xy_train = pd.merge(df, y_df, on='AccountID', how='inner')

    # Merge validation data with labels
    # xy_val = pd.merge(df_val, yval_df, on='AccountID', how='inner')

    # Extract features for training data
    feature_extractor1 = FraudDetectionFeatureExtractor(xtrain_df)
    df_fe = feature_extractor1.extract_all_features()
    df_fea2 = pd.merge(df, df_fe, on='AccountID', how='inner')
    df_fea = pd.merge(df_fea2, y_df, on='AccountID', how='inner')
    print("df_train_fea shape:", df_fea.shape)

    # Extract features for validation data
    feature_extractor2 = FraudDetectionFeatureExtractor(x_val_df)
    df_val_fe = feature_extractor2.extract_all_features()
    df_val_fea2 = pd.merge(df_val, df_val_fe, on='AccountID', how='inner')
    df_val_fea = pd.merge(df_val_fea2, yval_df, on='AccountID', how='inner')
    print("df_val_fea shape:", df_val_fea.shape)

    # Extract features for test data
    feature_extractor3 = FraudDetectionFeatureExtractor(test_df)
    df_test_fe = feature_extractor3.extract_all_features()
    df_test_fea = pd.merge(test_agg_df, df_test_fe, on='AccountID', how='inner')
    print("df_test_fea shape:", df_test_fea.shape)

    # Handle missing values in training data
    for column in df_fea.columns:
        if df_fea[column].isnull().sum() > 0:  # Check if the column has NaN values
            df_fea[column] = df_fea[column].fillna(df_fea[column].mean())

    # Handle missing values in validation data
    for column in df_val_fea.columns:
        if df_val_fea[column].isnull().sum() > 0:  # Check if the column has NaN values
            df_val_fea[column] = df_val_fea[column].fillna(df_val_fea[column].mean())

    # Handle missing values in test data
    for column in df_test_fea.columns:
        if df_test_fea[column].isnull().sum() > 0:  # Check if the column has NaN values
            df_test_fea[column] = df_test_fea[column].fillna(df_test_fea[column].mean())

    return df_fea, df_val_fea, df_test_fea

