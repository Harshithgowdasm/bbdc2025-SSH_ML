import pandas as pd
from datetime import timedelta

class FraudDetectionFeatureExtractor:
    def __init__(self, transaction_data):
        """
        Initialize the class with transaction data.
        :param transaction_data: DataFrame containing transaction data.
        """
        self.df = transaction_data

    def extract_time_based_features(self):
        """
        Extract time-based features from the transaction data.
        """
        # Convert 'Hour' to datetime (if not already)
        self.df['TransactionTime'] = pd.to_datetime(self.df['Hour'], unit='h')

        # Extract hour of the day
        self.df['HourOfDay'] = self.df['TransactionTime'].dt.hour

        # Extract day of the week
        self.df['DayOfWeek'] = self.df['TransactionTime'].dt.dayofweek  # Monday=0, Sunday=6

        # Time since last transaction (per account)
        self.df['TimeSinceLastTransaction'] = self.df.groupby('AccountID')['TransactionTime'].diff().dt.total_seconds() / 60  # In minutes

    def extract_balance_based_features(self):
        """
        Extract balance-based features from the transaction data.
        """
        # Balance Change Ratio
        self.df['BalanceChangeRatio'] = self.df['Amount'] / self.df['OldBalance']

        # Balance Deviation (from average balance per account)
        self.df['AverageBalance'] = self.df.groupby('AccountID')['NewBalance'].transform('mean')
        self.df['BalanceDeviation'] = self.df['NewBalance'] - self.df['AverageBalance']

    def extract_amount_based_features(self):
        """
        Extract transaction amount-based features from the transaction data.
        """
        # Transaction Amount Deviation
        self.df['AverageTransactionAmount'] = self.df.groupby('AccountID')['Amount'].transform('mean')
        self.df['TransactionAmountDeviation'] = self.df['Amount'] - self.df['AverageTransactionAmount']

        # Transaction Amount Ratio
        self.df['TransactionAmountRatio'] = self.df['Amount'] / self.df['AverageTransactionAmount']

    def extract_frequency_based_features(self):
        """
        Extract frequency-based features from the transaction data.
        """
        # Transaction Frequency (per Hour)
        self.df['TransactionFrequencyPerHour'] = self.df.groupby(['AccountID', 'Hour'])['Amount'].transform('count')

        # Transaction Burstiness (number of transactions within 10 minutes)
        # Sort the DataFrame by AccountID and TransactionTime
        self.df = self.df.sort_values(by=['AccountID', 'TransactionTime'])

        # Calculate the time difference between consecutive transactions
        self.df['TimeDiff'] = self.df.groupby('AccountID')['TransactionTime'].diff().dt.total_seconds() / 60  # In minutes

        # Flag transactions that occur within 10 minutes of the previous transaction
        self.df['IsBurst'] = (self.df['TimeDiff'] <= 10).astype(int)

        # Calculate the rolling count of transactions within 10 minutes
        self.df['TransactionBurstiness'] = self.df.groupby('AccountID')['IsBurst'].cumsum()

    def extract_behavioral_features(self):
        """
        Extract behavioral features from the transaction data.
        """
        # Transaction Type Distribution
        transaction_type_distribution = self.df.groupby(['AccountID', 'Action'])['Amount'].count().unstack(fill_value=0)
        transaction_type_distribution = transaction_type_distribution.div(transaction_type_distribution.sum(axis=1), axis=0)
        self.df = self.df.merge(transaction_type_distribution, on='AccountID', how='left')

        # Unusual Transaction Types (e.g., flag transactions of types that are <5% of the account's transactions)
        self.df['IsUnusualTransactionType'] = self.df.apply(lambda row: row[row['Action']] < 0.05, axis=1)

    def extract_external_account_features(self):
        """
        Extract external account-related features from the transaction data.
        """
        # External Account Flag
        self.df['IsExternalAccount'] = self.df['External'].notna().astype(int)

        # External Account Frequency
        self.df['ExternalAccountFrequency'] = self.df.groupby('External')['Amount'].transform('count')

    def extract_overdraft_features(self):
        """
        Extract overdraft-related features from the transaction data.
        """
        # Unauthorized Overdraft Flag (already in the data as 'isUnauthorizedOverdraft')

        # Overdraft Frequency
        self.df['OverdraftFrequency'] = self.df.groupby('AccountID')['isUnauthorizedOverdraft'].transform('sum')

    def extract_aggregated_features(self):
        """
        Extract aggregated features over time windows.
        """
        # Sort the DataFrame by AccountID and TransactionTime
        self.df = self.df.sort_values(by=['AccountID', 'TransactionTime'])

        # Rolling Average Transaction Amount (using a fixed-size window, e.g., 10 transactions)
        self.df['RollingAverageAmount'] = self.df.groupby('AccountID')['Amount'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())

        # Rolling Transaction Count (using a fixed-size window, e.g., 10 transactions)
        self.df['RollingTransactionCount'] = self.df.groupby('AccountID')['Amount'].transform(lambda x: x.rolling(window=10, min_periods=1).count())

    def aggregate_features(self):
        """
        Aggregate features at the account level.
        :return: DataFrame with aggregated features.
        """
        # Aggregate features at the account level
        aggregated_features = self.df.groupby('AccountID').agg({
            'HourOfDay': ['mean'],
            'DayOfWeek': ['mean'],
            'TimeSinceLastTransaction': ['mean'],
            'BalanceChangeRatio': ['mean'],
            'BalanceDeviation': ['mean'],
            'TransactionAmountDeviation': ['mean'],
            'TransactionAmountRatio': ['mean'],
            'TransactionFrequencyPerHour': ['mean'],
            'TransactionBurstiness': ['mean'],
            'IsUnusualTransactionType': ['sum'],
            'IsExternalAccount': ['sum'],
            'ExternalAccountFrequency': ['mean'],
            'isUnauthorizedOverdraft': ['sum'],
            'OverdraftFrequency': ['mean'],
            'RollingAverageAmount': ['mean'],
            'RollingTransactionCount': ['mean']
        })

        # Flatten the multi-level column index
        aggregated_features.columns = ['_'.join(col) for col in aggregated_features.columns]

        return aggregated_features

    def extract_all_features(self):
        """
        Extract all features and aggregate them at the account level.
        :return: DataFrame with aggregated features.
        """
        self.extract_time_based_features()
        self.extract_balance_based_features()
        self.extract_amount_based_features()
        self.extract_frequency_based_features()
        self.extract_behavioral_features()
        self.extract_external_account_features()
        self.extract_overdraft_features()
        self.extract_aggregated_features()

        # Aggregate features at the account level
        aggregated_features = self.aggregate_features()

        return aggregated_features


