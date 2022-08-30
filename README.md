# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2008 - 2021 seasons

## Installation
```bash
conda env create -f cfb_env.yaml
```

## Usage

```python
python cfb_ml.py
```
### Current prediction accuracies
```bash
Removed features (>=0.75 correlation):  ['rush_yds_per_att', 'first_down_pass', 'first_down_rush', 'penalty_yds', 'pass_int']

# Best hyperparameters - classification
=======
GradientBoostingClassifier - best params:  {'criterion': 'friedman_mse', 'learning_rate': 0.30000000000000004, 'loss': 'log_loss', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 300}
RandomForestClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 300}
DecisionTreeClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'splitter': 'best'}
AdaClassifier - best params:  {'algorithm': 'SAMME.R', 'learning_rate': 1.0, 'n_estimators': 150}
LogisticRegression - best params: {'C': 1.5, 'max_iter': 800, 'penalty': 'l2', 'solver': 'lbfgs'}
MLPClassifier - best params:  {'learning_rate': 'invscaling', 'learning_rate_init': 0.004, 'max_iter': 700, 'solver': 'lbfgs'}
KNeighborsClassifier - best params:  {'algorithm': 'auto', 'n_neighbors': 100, 'p': 1, 'weights': 'distance'}
XGBBoost Classifier - best params: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 180}

#Best hyperparameters - regression
======
'GradientBoosting': {'criterion': 'squared_error', 'learning_rate': 0.4, 'loss': 'absolute_error', 'max_depth': 1, 'max_features': 'log2', 'n_estimators': 300}
'DecisionTree':  {'criterion': 'squared_error', 'max_features': 'log2', 'min_samples_split': 4, 'splitter': 'random'}
'RandomForest': {'criterion': 'absolute_error', 'max_features': 'sqrt', 'min_samples_split': 4, 'n_estimators': 400}
'Ada':  {'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 50}
'LinearRegression': {'C': 3.5, 'max_iter': 600, 'penalty': 'l2', 'solver': 'lbfgs'}
'KNearestNeighbor': {'n_neighbors': 25, 'weights':'distance','algorithm': 'brute','p': 2}
'MLP':  {'activation': 'identity', 'learning_rate': 'invscaling', 'learning_rate_init': 0.002, 'max_iter': 500, 'solver': 'adam'}
'XGB-boost': {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 60}

# Classification accuracy
=======
GradientBoostingClassifier accuracy 0.8042639593908629
RandomForestClassifier accuracy 0.7835532994923858
DecisionTreeClassifier accuracy 0.743756345177665
AdaClassifier accuracy 0.8036548223350254
LogisticRegression  accuracy 0.8093401015228426
MLPClassifier accuracy 0.8081218274111676
KNeighborsClassifier accuracy 0.7474111675126903
XGBClassifier accuracy 0.8048730964467005
KerasClassifier: test loss, test acc: [0.6931477785110474, 0.5005075931549072]
#Keras hyperparams: optimizer='SGD' or 'Adam,loss='binary_crossentropy'

#Regression explained variance - R^2
GradientBoostingRegressor accuracy 0.859226834130618
RandomForestRegressor accuracy 0.8529285888408278
DecisionTreeRegressor accuracy 0.7068270393345268
AdaRegressor accuracy 0.7697904471677975
LinearRegression  accuracy 0.8669813533143526
MLPRegressor accuracy 0.8657863603262631
KNeighborsRegressor accuracy 0.42648046641011716
XGBRegressor accuracy 0.8561383119974133
KerasRegression accuracy  0.8642884018447907
#Regression RMSE
GradientBoostingRegressor rmse 4.864591076853353
RandomForestRegressor rmse 4.972222217335514
DecisionTreeRegressor rmse 7.020181757174267
AdaRegressor rmse 6.22082623780695
LinearRegression  rmse 4.7287098031777335
MLPRegressor rmse 4.749902849189487
KNeighborsRegressor rmse 9.818847324147438
XGBRegressor rmse 4.917665401026602
KerasRegression rmse  4.7763361320934345
=======

check the amount of wins and losses are in the training label data (should be almost equal):
1    12191
0    11005

```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)


### Feature Importances Classification
![](https://github.com/bszek213/college_football_machine_learning/blob/master/Classification/FeatureImportance.png)

### Feature Importances Regression
![](https://github.com/bszek213/college_football_machine_learning/blob/master/Regression/FeatureImportance.png)

### Keras Classification Epochs
![](https://github.com/bszek213/college_football_machine_learning/blob/master/Classification/keras_model_acc.png)

### Keras Regression Epochs
![](https://github.com/bszek213/college_football_machine_learning/blob/master/Regression/keras_model_regression.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
