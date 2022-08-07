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
GradientBoostingClassifier - best params:  {'criterion': 'squared_error', 'learning_rate': 0.4, 'loss': 'log_loss', 'max_depth': 1, 'max_features': 'log2', 'n_estimators': 400}
RandomForestClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 100}
DecisionTreeClassifier - best params:  {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'log2', 'splitter': 'best'}
AdaClassifier - best params:  {'algorithm': 'SAMME', 'learning_rate': 1.5, 'n_estimators': 150}
LogisticRegression - best params: {'C': 1.5, 'max_iter': 900, 'penalty': 'l2', 'solver': 'lbfgs'}
MLPClassifier - best params:  GridSearchCV(estimator=MLPClassifier(), n_jobs=-1,
             param_grid={'learning_rate': ['constant', 'invscaling',
                                           'adaptive'],
                         'learning_rate_init': array([0.001, 0.002, 0.003, 0.004]),
                         'max_iter': range(100, 1000, 200),
                         'solver': ['lbfgs', 'sgd', 'adam']},
             refit='accuracy', scoring=['accuracy'], verbose=4)

KNeighborsClassifier - best params:  {'algorithm': 'ball_tree', 'n_neighbors': 100, 'p': 1, 'weights': 'distance'}
GradientBoostingClassifier accuracy 0.788495891389782
RandomForestClassifier accuracy 0.7631296891747053
DecisionTreeClassifier accuracy 0.7281171847088246
AdaClassifier accuracy 0.7906395141121829
LogisticRegression  accuracy 0.7938549481957842
MLPClassifier accuracy 0.7981421936405859
KNeighborsClassifier accuracy 0.7220435869953555
KerasClassifier accuracy 0.7827577253182729

check the amount of wins and losses are in the training label data:
wins    5884
losses    5312

```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
