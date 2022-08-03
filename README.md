# College Football Game Predictions

Scripts that predict the outcome of any Division I college football game. Data are from 2008 - 2021 seasons

## Installation
```bash
conda env create -f cfb_env
```

## Usage

```python
python cfb_ml.py
```
### Current prediction accuracies
```bash
Removed features (>0.90 correlation):  ['penalty_yds']
# Best params for each model
GradientBoostingClassifier - best params:  {'criterion': 'squared_error', 'learning_rate': 0.4, 'loss': 'deviance', 'max_depth': 1, 'max_features': 'auto', 'n_estimators': 400}
RandomForestClassifier - best params:  {'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 100}
DecisionTreeClassifier - best params:  {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'splitter': 'best'}
SVC - best params:  {'C': 4.5, 'gamma': 'scale', 'kernel': 'linear', 'tol': 0.008}
LogisticRegression - best params: {'C': 4.0, 'max_iter': 300, 'penalty': 'l2', 'solver': 'lbfgs'}
MLPClassifier - best params:  GridSearchCV(cv=5, estimator=MLPClassifier(), n_jobs=-1,
             param_grid={'learning_rate': ['constant', 'invscaling',
                                           'adaptive'],
                         'learning_rate_init': array([0.001, 0.002, 0.003, 0.004]),
                         'max_iter': range(100, 1000, 100),
                         'solver': ['lbfgs', 'sgd', 'adam'],
                         'tol': array([0.001, 0.002, 0.003, 0.004])},
             refit='accuracy', scoring=['accuracy'], verbose=4)
KNeighborsClassifier - best params:  {'algorithm': 'auto', 'n_neighbors': 100, 'p': 1, 'weights': 'distance'}
#Prediction accuracies
Gradclass 0.7983643542019176
RandForclass 0.7710095882684715
DecTreeclass 0.739424703891709
SVCclass 0.8056965595036661
LogReg 0.8040045121263395
MLPClass 0.81443880428652
KClass 0.7848279751833052
```
### Correlation Matrix
![](https://github.com/bszek213/college_football_machine_learning/blob/master/correlations.png)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.