# College Football Game Predictions

Scripts that predict the outcome of any Division I college football game. Data are from 2008 - 2021 seasons

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
Removed features (>=0.90 correlation):  ['penalty_yds']
GradientBoostingClassifier - best params:  {'criterion': 'friedman_mse', 'learning_rate': 0.2, 'loss': 'log_loss', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 400}
RandomForestClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 200}
DecisionTreeClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'splitter': 'best'}
LogisticRegression - best params: {'C': 1.5, 'max_iter': 700, 'penalty': 'l2', 'solver': 'lbfgs'}
MLPClassifier - best params:  GridSearchCV(estimator=MLPClassifier(), n_jobs=-1,
             param_grid={'learning_rate': ['constant', 'invscaling',
                                           'adaptive'],
                         'learning_rate_init': array([0.001, 0.002, 0.003, 0.004]),
                         'max_iter': range(100, 1000, 200),
                         'solver': ['lbfgs', 'sgd', 'adam']},
             refit='accuracy', scoring=['accuracy'], verbose=4)
KNeighborsClassifier - best params:  {'algorithm': 'kd_tree', 'n_neighbors': 100, 'p': 1, 'weights': 'distance'}
GradientBoostingClassifier accuracy 0.789210432297249
RandomForestClassifier accuracy 0.7642015005359056
DecisionTreeClassifier accuracy 0.7266881028938906
LogisticRegression  accuracy 0.797427652733119
MLPClassifier accuracy 0.7995712754555199
KNeighborsClassifier accuracy 0.7252590210789568
KerasClassifier accuracy 0.7756058268430757

```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
