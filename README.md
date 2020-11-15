# FakeNewsClassifier
Predict whether news is fake or not
Brief steps about Fake News Classification model:
1. Fetched training and testing data
2. Checked for null data and found that all fake (labelled 1) label is null in training and testing set
3. Impute fake title with "Fake Fake Fake" value
4. Checked for imbalance labels using countplot and found that no imbalance data in training set
5. Created analyzer function to convert "title" to remove non alphabetic values and stopwords. As well as lower the sentence
6. Created one hot representation and pre-padding with vocabulary size of 5000,sentence lenght to 50 and dimention 40
7. Create function to select best model with different set of hyper parameter and different layers and drop out
8. Did hyper parameter tuning using GridSerachCV with different set of parameters to evaluate word embedded LSTM model.
9. Trained model using best parameter obtained
10. After following all pre-processing steps performed prediction on test.csv.
11. Optinal: merged the original submission.csv with test.csv and tried to find confusion matrix and classification report.
12. Saved my predicted output in "my_submission.csv"

Note:
1. FakeClassifierWithRNNandHyperParameterTuning.ipynb is working model for which above steps are
