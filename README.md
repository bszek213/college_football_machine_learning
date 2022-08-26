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
GradientBoostingClassifier - best params:  {'criterion': 'friedman_mse', 'learning_rate': 0.30000000000000004, 'loss': 'log_loss', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 300}
RandomForestClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 300}
DecisionTreeClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'splitter': 'best'}
AdaClassifier - best params:  {'algorithm': 'SAMME.R', 'learning_rate': 1.0, 'n_estimators': 150}
LogisticRegression - best params: {'C': 1.5, 'max_iter': 800, 'penalty': 'l2', 'solver': 'lbfgs'}
MLPClassifier - best params:  {'learning_rate': 'invscaling', 'learning_rate_init': 0.004, 'max_iter': 700, 'solver': 'lbfgs'}
KNeighborsClassifier - best params:  {'algorithm': 'auto', 'n_neighbors': 100, 'p': 1, 'weights': 'distance'}
XGBBoost Classifier - best params: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 180}
# Classification accuracy
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
GradientBoostingRegressor accuracy 0.8699625600882974
RandomForestRegressor accuracy 0.9457682994612451
DecisionTreeRegressor accuracy 0.894882444151152
AdaRegressor accuracy 0.7708454126882668
LinearRegression  accuracy 0.8752138410893351
MLPRegressor accuracy 0.8679400665500603
KNeighborsRegressor accuracy 0.8790189462394215
XGBRegressor accuracy 0.8718155201012868
KerasRegression accuracy  0.8725287425688582
#Regression RMSE
GradientBoostingRegressor rmse 4.751835272714411
RandomForestRegressor rmse 3.0686954044737393
DecisionTreeRegressor rmse 4.2723300782230496
AdaRegressor rmse 6.307996582043323
LinearRegression  rmse 4.654900260950782
MLPRegressor rmse 4.788645730183061
KNeighborsRegressor rmse 4.583379870209777
XGBRegressor rmse 4.717858317038032
KerasRegression rmse  4.704714853031657

check the amount of wins and losses are in the training label data (should be almost equal):
1    12191
0    11005

```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)

### Feature Importances Classification
![](https://github.com/bszek213/college_football_machine_learning/blob/master/FeatureImportance.png)

### Feature Importances Regression
![](https://github.com/bszek213/college_football_machine_learning/blob/regression/Regression/FeatureImportance.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
