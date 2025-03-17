from autogluon.tabular import TabularPredictor, TabularDataset

predictor = TabularPredictor.load("bbdc/AutogluonModels/ag-20250317_143431") #TODO: provide the path 
test_data = TabularDataset('test_set/x_test_aggregated.csv')

y_pred = predictor.predict(test_data)

#write to the predictions as columns in the test set
test_data['Fraudster'] = y_pred
test_data.to_csv('test_set/x_test_aggregated_predictions.csv', index=False)
print(test_data.head())