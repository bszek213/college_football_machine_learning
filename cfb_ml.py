# -*- coding: utf-8 -*-
"""
College football game predictor
"""
from html_parse_cfb import html_to_df_web_scrape
import argparse
from sportsipy.ncaaf.teams import Teams
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from os import getcwd
from os.path import join, exists
class cfb:
    def __init__(self):
        print('initialize class cfb')
        self.all_data = pd.DataFrame()
    def input_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-t1", "--team1", help = "team 1 input")
        parser.add_argument("-t2", "--team2", help = "team 2 input")
        parser.add_argument("-g", "--games", help = "number of games for test data")
        self.args = parser.parse_args()
    def get_teams(self):
        final_dir = join(getcwd(), 'all_data.csv')
        isExists = exists(final_dir)
        if isExists == False:
            year_list = [2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010]
            for year in year_list:
                all_teams = Teams(year)
                team_names = all_teams.dataframes.abbreviation
                print(team_names)
                final_list = []
                self.year_store = year
                for abv in team_names:
                    print(f'current team: {abv}, year: {year}')
                    # team = all_teams(abv)
                    str_combine = 'https://www.sports-reference.com/cfb/schools/' + abv.lower() + '/' + str(self.year_store) + '/gamelog/'
                    df_inst = html_to_df_web_scrape(str_combine)
                    final_list.append(df_inst)
                output = pd.concat(final_list)
                output['game_result'].loc[output['game_result'].str.contains('W')] = 'W'
                output['game_result'].loc[output['game_result'].str.contains('L')] = 'L'
                output['game_result'] = output['game_result'].replace({'W': 1, 'L': 0})
                final_data = output.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
                self.all_data = pd.concat([self.all_data, final_data.dropna()])
                print('len data: ', len(self.all_data))
            self.all_data.to_csv('all_data.csv')
        else:
            self.all_data = pd.read_csv(final_dir)
    def split(self):
        self.y = self.all_data['game_result']
        self.x = self.all_data.drop(columns=['game_result'])
        self.correlate_analysis()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y, train_size=0.8)

    def correlate_analysis(self):
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find features with correlation greater than 0.85
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.85)]
        # print('drop these:', to_drop)
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)

        #Create new scaled data - DO I REMOVE THE VARIABLES THAT ARE HIGHLY 
        # CORRELATED BEFORE I STANDARDIZE THEM OR STANDARDIZE AND THEN REMOVE
        # HIGHLY CORRELATED
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.x_no_corr)
        cols = self.x_no_corr.columns
        self.x = pd.DataFrame(scaled_data, columns = cols)
        
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        #plot heat map
        g=sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
        plt.savefig('correlations.png')
        plt.close()

    def machine(self):
        Gradclass = GradientBoostingClassifier()
        Grad_perm = {
            'loss' : ['deviance', 'exponential'],
            'learning_rate': np.arange(0.1, .5, 0.1, dtype=float),
            'n_estimators': range(100,500,100),
            'criterion' : ['friedman_mse', 'squared_error'],
            'max_depth': np.arange(1, 5, 1, dtype=int),
            'max_features' : ['auto', 'sqrt', 'log2']
            }
        clf = GridSearchCV(Gradclass, Grad_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_Grad = clf.fit(self.x_train,self.y_train)
        
        RandForclass = RandomForestClassifier()
        Rand_perm = {
            'criterion' : ["gini", "entropy"],
            'n_estimators': range(100,500,100),
            'max_depth': np.arange(1, 5, 1, dtype=int),
            'max_features' : ['auto', 'sqrt', 'log2']
            }
        clf_rand = GridSearchCV(RandForclass, Rand_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_rand = clf_rand.fit(self.x_train,self.y_train)
        # RandForclass.fit(self.x_train,self.y_train)
        
        DecTreeclass = DecisionTreeClassifier()
        Dec_perm = {
            'splitter' : ["best", "random"],
            'criterion' : ["gini", "entropy"],
            'max_depth': np.arange(1, 5, 1, dtype=int),
            'max_features' : ['auto', 'sqrt', 'log2']
            }
        clf_dec = GridSearchCV(DecTreeclass, Dec_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_dec = clf_dec.fit(self.x_train,self.y_train)
        
        # DecTreeclass.fit(self.x_train,self.y_train)
        
        SVCclass = SVC()
        SVC_perm = {
            'C': np.arange(1, 5, 0.5, dtype=float),
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma' : ['scale', 'auto'],
            'tol': np.arange(0.001, 0.01, 0.001,dtype=float),
            }
        clf_SVC = GridSearchCV(SVCclass, SVC_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_SVC = clf_SVC.fit(self.x_train,self.y_train)
        # SVCclass.fit(self.x_train,self.y_train)
        
        LogReg = LogisticRegression()
        log_reg_perm = {
            'penalty': ['l2'],
            'C': np.arange(1, 5, 0.5, dtype=float),
            'max_iter': range(100,1000,100),
            'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
            }
        clf_Log = GridSearchCV(LogReg, log_reg_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_Log = clf_Log.fit(self.x_train,self.y_train)
        
        MLPClass = MLPClassifier()
        MLP_perm = {
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'learning_rate' : ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init' : np.arange(0.001, 0.005, 0.001, dtype=float),
            'max_iter': range(100,1000,100),
            'tol': np.arange(0.001, 0.005, 0.001, dtype=float)
            }
        clf_MLP = GridSearchCV(MLPClass, MLP_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_MLP= clf_MLP.fit(self.x_train,self.y_train)
        # MLPClass.fit(self.x_train,self.y_train)
        
        KClass = KNeighborsClassifier()
        KClass_perm = {
            'n_neighbors' : range(100,1000,100),
            'weights' : ['uniform', 'distance'],
            'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p' : [1,2]
            }
        clf_KClass = GridSearchCV(KClass, KClass_perm, scoring=['accuracy'],
                           refit='accuracy',cv=5, verbose=4)
        search_KClass= clf_KClass.fit(self.x_train,self.y_train)
        # KClass.fit(self.x_train,self.y_train)
        
        # PerClass = Perceptron() #Terrible model for these data
        # PerClass.fit(self.x_train,self.y_train)
        
        Gradclass_err = accuracy_score(self.y_test, search_Grad.predict(self.x_test))
        RandForclass_err = accuracy_score(self.y_test, search_rand.predict(self.x_test))
        DecTreeclass_err = accuracy_score(self.y_test, search_dec.predict(self.x_test))
        SVCclass_err = accuracy_score(self.y_test, search_SVC.predict(self.x_test))
        LogReg_err = accuracy_score(self.y_test, search_Log.predict(self.x_test))
        MLPClass_err = accuracy_score(self.y_test, search_MLP.predict(self.x_test))
        KClass_err = accuracy_score(self.y_test, search_KClass.predict(self.x_test))
        # PerClass_err = accuracy_score(self.y_test, PerClass.predict(self.x_test))

        print('Removed features (>0.85 correlation): ', self.drop_cols)
        print('GradientBoostingClassifier - best params: ',search_Grad.best_params_)
        print('RandomForestClassifier - best params: ',search_rand)
        print('DecisionTreeClassifier - best params: ',search_dec)
        print('SVC - best params: ',search_SVC)
        print('LogisticRegression - best params:',search_Log.best_params_)
        print('MLPClassifier - best params: ',search_MLP)
        print('KNeighborsClassifier - best params: ',search_KClass)
        print('Gradclass',Gradclass_err)
        print('RandForclass',RandForclass_err)
        print('DecTreeclass',DecTreeclass_err)
        print('SVCclass',SVCclass_err)
        print('LogReg',LogReg_err)
        print('MLPClass',MLPClass_err)
        print('KClass',KClass_err)
        # print('PerClass',PerClass_err)


def main():
    start_time = time.time()
    cfb_class = cfb()
    cfb_class.input_arg()
    cfb_class.get_teams()
    cfb_class.split()
    cfb_class.machine()
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main()
    
