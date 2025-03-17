from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('merged_training_data.csv')
train_data.head()

label = 'Fraudster'
#print(train_data[label].describe())
predictor = TabularPredictor(label=label).fit(train_data)

test_data = TabularDataset('merged_validation_data.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))

predictor.evaluate(test_data, silent=True)
print(predictor.leaderboard(test_data))