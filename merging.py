import pandas as pd

train_data = pd.read_csv('validation_set/x_val_aggregated.csv')

y_label = pd.read_csv('validation_set/y_val.csv')

merged_df = pd.merge(train_data, y_label, on='AccountID')
merged_df.to_csv('merged_validation_data.csv', index=False)