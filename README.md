# College Football Game Predictions

machine learning that predicts the outcome of any Division I college football game. Data are from 2000 - 2022 seasons.
Current accuracy is 77.96% across 246 games in 2022.

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
Removed features (>=0.8 correlation):  ['rush_yds_per_att', 'first_down_pass', 'first_down_rush', 'penalty_yds']

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
'GradientBoosting': {'criterion': 'friedman_mse', 'learning_rate': 0.4, 'loss': 'squared_error', 'max_depth': 1, 'max_features': 'sqrt', 'n_estimators': 400}
'DecisionTree':  {'criterion': 'squared_error', 'max_features': 'log2', 'min_samples_split': 4, 'splitter': 'random'}
'SVM':  {'C': 2.5, 'degree': 1.0, 'gamma': 'scale', 'kernel': 'linear', 'tol': 0.001}
'RandomForest': {'criterion': 'squared_error', 'max_features': 'sqrt', 'min_samples_split': 3, 'n_estimators': 300}
'Ada':  {'learning_rate': 0.5, 'loss': 'exponential', 'n_estimators': 50}
'KNearestNeighbor': {'algorithm': 'auto', 'n_neighbors': 30, 'p': 1, 'weights': 'distance'}
'MLP':   {'activation': 'identity', 'learning_rate': 'invscaling', 'learning_rate_init': 0.002, 'max_iter': 900, 'solver': 'lbfgs'}
'XGB-boost': {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 180}
'PassiveAggressiveRegressor' : {C': 1.0, 'epsilon': 0.30000000000000004, 'max_iter': 500, 'tol': 0.010000000000000002}

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

# Regression explained variance - R^2
GradientBoostingRegressor accuracy 0.9081065771017075
SVM accuracy 0.9001070575828227
RandomForestRegressor accuracy 0.9610997140185163
DecisionTreeRegressor accuracy 0.9094600165356903
AdaRegressor accuracy 0.8316380699858044
LinearRegression  accuracy 0.9037894548841173
MLPRegressor accuracy 0.903872588181907
KNeighborsRegressor accuracy 0.9087505893124272
XGBRegressor accuracy 0.9097280310555114
KerasRegression accuracy  0.8997792797235109
PassiveAggressive accuracy  0.9025627255172579
====================================
Model with the highest R^2: RandomForest
====================================
# Regression RMSE
GradientBoostingRegressor rmse 4.254611840428275
SVM rmse 4.435934582747479
RandomForestRegressor rmse 2.768178728704508
DecisionTreeRegressor rmse 4.223163888820945
AdaRegressor rmse 5.758901863661174
LinearRegression  rmse 4.353404959676689
MLPRegressor rmse 4.35152371513948
KNeighborsRegressor rmse 4.239676932806324
XGBRegressor rmse 4.216908596549564
KerasRegression rmse  4.443206419520521
PassiveAggressive rmse  4.431370005092532
====================================
Model with the lowest RMSE: RandomForest
====================================
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
