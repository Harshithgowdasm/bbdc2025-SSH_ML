import pandas as pd

df1 = pd.read_csv('student_skeleton.csv')
df2 = pd.read_csv('test_set/x_test_aggregated_predictions.csv')

fraudster_map = df2.set_index('AccountID')['Fraudster'].to_dict()

#match and write the predictions
df1['Fraudster'] = df1.apply(
    lambda row: fraudster_map.get(row['AccountID'], row['Fraudster']) if pd.isna(row['Fraudster']) or row['Fraudster'] == '' else row['Fraudster'], 
    axis=1
)

df1.to_csv('student_skeleton.csv', index=False)
