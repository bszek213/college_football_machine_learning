#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
College football game regressor
@author: brianszekely
"""
from html_parse_cfb import html_to_df_web_scrape
import argparse
from sportsipy.ncaaf.teams import Teams
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score #explained_variance_score
import time
from sklearn.model_selection import GridSearchCV
# from scipy.stats import uniform
from os import getcwd
from os.path import join, exists
from scipy import stats
# from keras.utils import np_utils
# for modeling
from keras.models import Sequential
from keras.layers import Dense#, Dropout
# from keras.callbacks import EarlyStopping
import yaml
from tensorflow.keras.metrics import RootMeanSquaredError
# import tensorflow as tf
import xgboost as xgb
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
# from time import sleep
#TODO: Build the keras hyperparam tuner
class cfb_regressor():
    def __init__(self):
        print('initialize class cfb')
        self.all_data = pd.DataFrame()
    def read_hyper_params(self):
        final_dir = join(getcwd(), 'hyper_params_regress.yaml')
        isExists = exists(final_dir)
        if isExists == True:
            with open(final_dir) as file:
                self.hyper_param_dict = yaml.load(file, Loader=yaml.FullLoader)

    def get_teams(self):
        final_dir = join(getcwd(), 'all_data_regressor.csv')
        print(final_dir)
        isExists = exists(final_dir)
        print(isExists)
        if isExists == False:
            year_list = [2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
            for year in year_list:
                all_teams = Teams(year)
                team_names = all_teams.dataframes.abbreviation
                team_names = team_names.sort_values()
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
                output['game_result'] = output['game_result'].str.replace('W','')
                output['game_result'] = output['game_result'].str.replace('L','')
                output['game_result'] = output['game_result'].str.replace('(','')
                output['game_result'] = output['game_result'].str.replace(')','')
                output['game_result'] = output['game_result'].str.split('-').str[0]
                output['game_result'] = output['game_result'].str.replace('-','')
                final_data = output.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
                self.all_data = pd.concat([self.all_data, final_data.dropna()])
                print('len data: ', len(self.all_data))
            self.all_data.to_csv('all_data_regresso.csv')
        else:
            self.all_data = pd.read_csv(final_dir)
    def split(self):
        self.y = self.all_data['game_result']
        self.x = self.all_data.drop(columns=['game_result'])
        self.pre_process()

    def pre_process(self):
        #drop irrelavent column
        if 'Unnamed: 0' in self.x.columns:
            self.x = self.x.drop(columns=['Unnamed: 0'])
        
        # Find features with correlation greater than 0.90
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.75)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        cols = self.x_no_corr.columns
        
        #Remove outliers with 1.5 +/- IQR
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 25)
            Q3 = np.percentile(self.x_no_corr[col_name], 75)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+2.0*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-2.0*IQR)) 
            self.x_no_corr.drop(upper[0], inplace = True)
            self.x_no_corr.drop(lower[0], inplace = True)
            self.y.drop(upper[0], inplace = True)
            self.y.drop(lower[0], inplace = True)
            if 'level_0' in self.x_no_corr.columns:
                self.x_no_corr.drop(columns=['level_0'],inplace = True)
            self.x_no_corr.reset_index(inplace = True)
            self.y.reset_index(inplace = True, drop=True)
        self.x_no_corr.drop(columns=['level_0','index'],inplace = True)
        print(f'new feature dataframe shape after outlier removal: {self.x_no_corr.shape}')

        #split data into train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr,self.y, train_size=0.8)
        for col_name in cols:
            # self.x_train[col_name], _ = stats.boxcox(self.x_train[col_name])
            self.prob_plots(col_name)
        #plot heat map
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        g=sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
        plt.savefig('correlations.png')
        plt.close()
    def prob_plots(self,col_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        prob = stats.probplot(self.x_train[col_name], dist=stats.norm, plot=ax1)
        title = f'probPlot of training data against normal distribution, feature: {col_name}'  
        ax1.set_title(title,fontsize=10)
        save_name = 'probplot_' + col_name + '.png'
        plt.tight_layout()
        plt.savefig(join(getcwd(), 'prob_plots_regress',save_name), dpi=200)
    def machine(self):
        #Drop data that poorly fit the normally distribution
        # self.x_train.drop(columns=['turnovers','first_down_penalty','fumbles_lost'], inplace=True)
        # self.x_test.drop(columns=['turnovers','first_down_penalty','fumbles_lost'], inplace=True)
        #load in the hyperparams from file if the file exists
        final_dir = join(getcwd(), 'hyper_params_regress.yaml')
        isExists = exists(final_dir)
        if isExists == True:
            print('Found yaml - reading in hyperparameters now and fitting')
            Gradclass = GradientBoostingRegressor(**self.hyper_param_dict['GradientBoosting']).fit(self.x_train,self.y_train)
            RandForclass = RandomForestRegressor(**self.hyper_param_dict['RandomForest']).fit(self.x_train,self.y_train)
            ada_class = AdaBoostRegressor(**self.hyper_param_dict['Ada']).fit(self.x_train,self.y_train)
            DecTreeclass = DecisionTreeRegressor(**self.hyper_param_dict['DecisionTree']).fit(self.x_train,self.y_train)
            LinReg = LinearRegression().fit(self.x_train,self.y_train)
            KClass = KNeighborsRegressor(**self.hyper_param_dict['KNearestNeighbor']).fit(self.x_train,self.y_train)
            MLPClass = MLPRegressor(**self.hyper_param_dict['MLP']).fit(self.x_train,self.y_train)
            xgb_class = xgb.XGBRegressor(**self.hyper_param_dict['XGB-boost']).fit(self.x_train,self.y_train)  
            #Keras classifier 
            model = Sequential()
            # model.add(LSTM(12))
            model.add(Dense(12, input_shape=(self.x_train.shape[1],), activation="relu"))#input shape - (features,)
            # model.add(Dropout(0.3))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(12, activation='softmax'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.summary() 
            #compile 
            model.compile(optimizer='SGD', 
                  loss='mse',
                  metrics=[RootMeanSquaredError()])
            history = model.fit(self.x_train,
                        self.y_train,
                        # callbacks=[es],
                        epochs=200, # you can set this to a big number!
                        batch_size=20,
                        validation_split=0.2,           
                        # validation_data=(self.x_test, self.y_test),
                        shuffle=True,
                        verbose=1)
            # keras_acc = history.history['accuracy']
            # pred_train = history.predict(self.x_test) #will need this in the future when I want to look at one team vs. another
            scores = model.evaluate(self.x_test, self.y_test, verbose=0)
            keras_y_predict = model.predict(self.x_test)
            plt.figure()
            plt.plot(history.history['root_mean_squared_error'])
            plt.plot(history.history['val_root_mean_squared_error'])
            plt.title('Keras Regression')
            plt.xlabel('Epochs')
            plt.ylabel('Square Root Mean Squared error')
            plt.legend(['train','test'])
            save_name = 'keras_model_regression' + '.png'
            plt.savefig(join(getcwd(),save_name), dpi=200)
            plt.close()
        else:
            Gradclass = GradientBoostingRegressor()
            Grad_perm = {
                'loss' : ['squared_error', 'absolute_error'],
                'learning_rate': np.arange(0.1, .5, 0.1, dtype=float),
                'n_estimators': range(100,500,100),
                'criterion' : ['friedman_mse', 'squared_error'],
                'max_depth': np.arange(1, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            clf = GridSearchCV(Gradclass, Grad_perm, scoring=['neg_mean_absolute_error'],
                                refit='neg_mean_absolute_error', verbose=4, n_jobs=-1)
            search_Grad = clf.fit(self.x_train,self.y_train)
            RandForclass = RandomForestRegressor()
            Rand_perm = {
                'criterion' : ["squared_error", "absolute_error"],
                'n_estimators': range(100,500,100),
                'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            clf_rand = GridSearchCV(RandForclass, Rand_perm, scoring=['neg_mean_absolute_error'],
                               refit='neg_mean_absolute_error',verbose=4, n_jobs=-1)
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            DecTreeclass = DecisionTreeRegressor()
            Dec_perm = {
                'splitter' : ["best", "random"],
                'criterion' : ["squared_error", "friedman_mse"],
                'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            clf_dec = GridSearchCV(DecTreeclass, Dec_perm, scoring=['neg_mean_absolute_error'],
                               refit='neg_mean_absolute_error',verbose=4, n_jobs=-1)
            search_dec = clf_dec.fit(self.x_train,self.y_train)
            ada_class = AdaBoostRegressor()
            ada_perm = {'n_estimators': range(50,200,50),
                          'learning_rate': np.arange(.5,2.5,.5,dtype=float),
                          'loss': ['linear','square','exponential']}
            clf_ada = GridSearchCV(ada_class, ada_perm, scoring=['neg_mean_absolute_error'],
                                refit='neg_mean_absolute_error', verbose=4, n_jobs=-1)
            search_ada = clf_ada.fit(self.x_train,self.y_train)
            LinReg = LinearRegression()
            search_Ling = LinReg.fit(self.x_train,self.y_train)
            
            MLPClass = MLPRegressor()
            MLP_perm = {
                'activation':['identity','relu','tanh'],
                'solver' : ['lbfgs', 'sgd', 'adam'],
                'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init' : np.arange(0.001, 0.005, 0.001, dtype=float),
                'max_iter': range(100,1000,200),
                # 'tol': np.arange(0.001, 0.005, 0.001, dtype=float)
                }
            clf_MLP = GridSearchCV(MLPClass, MLP_perm, scoring=['neg_mean_absolute_error'],
                               refit='neg_mean_absolute_error', verbose=4, n_jobs=-1)
            search_MLP= clf_MLP.fit(self.x_train,self.y_train)
            KClass = KNeighborsRegressor()
            KClass_perm = {
                'n_neighbors' : np.arange(5,100,25),
                'weights' : ['uniform', 'distance'],
                'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p' : [1,2]
                }
            clf_KClass = GridSearchCV(KClass, KClass_perm, scoring=['neg_mean_absolute_error'],
                               refit='neg_mean_absolute_error', verbose=4, n_jobs=-1)
            search_KClass= clf_KClass.fit(self.x_train,self.y_train)
            estimator = xgb.XGBRegressor(
                    nthread=4,
                    seed=42
                    )
            parameters_xgb = {
                        'max_depth': range (2, 10, 1),
                        'n_estimators': range(60, 220, 40),
                        'learning_rate': [0.1, 0.01, 0.05]
                        }
            grid_search_xgb = GridSearchCV(
                                        estimator=estimator,
                                        param_grid=parameters_xgb,
                                        scoring = 'neg_mean_absolute_error',
                                        n_jobs = -1,
                                        cv = 5,
                                        verbose=4
                                        )
    
            grid_search_xgb.fit(self.x_train,self.y_train)
        print('Removed features (>=0.75 correlation): ', self.drop_cols)
        if isExists == False:
            print('GradientBoostingRegressor - best params: ',search_Grad.best_params_)
            print('RandomForestRegressor - best params: ',search_rand.best_params_)
            print('DecisionTreeRegressor- best params: ',search_dec.best_params_)
            # print('SVC - best params: ',search_SVC.best_params_)
            print('AdaRegressor - best params: ',search_ada.best_params_)
            # print('LinearRegression- best params:',search_Ling.best_params_)
            print('MLPRegressor - best params: ',search_MLP.best_params_)
            print('KNeighborsRegressor- best params: ',search_KClass.best_params_)
            print('XGB-boost - best params: ',grid_search_xgb.best_params_)
            return 'no model'
        else:
            Gradclass_err = r2_score(self.y_test, Gradclass.predict(self.x_test))
            RandForclass_err = r2_score(self.y_test, RandForclass.predict(self.x_test))
            DecTreeclass_err = r2_score(self.y_test, DecTreeclass.predict(self.x_test))
            # SVCclass_err = accuracy_score(self.y_test, search_SVC.predict(self.x_test))
            adaclass_err = r2_score(self.y_test, ada_class.predict(self.x_test))
            LinReg_err = r2_score(self.y_test, LinReg.predict(self.x_test))
            MLPClass_err = r2_score(self.y_test, MLPClass.predict(self.x_test))
            KClass_err = r2_score(self.y_test, KClass.predict(self.x_test))
            XGB_err = r2_score(self.y_test, xgb_class.predict(self.x_test))
            keras_err = r2_score(self.y_test, keras_y_predict)
            # print(f'Keras best params: {keras_grid.best_score_}, {keras_grid.best_params_}')
            print('GradientBoostingRegressor accuracy',Gradclass_err)
            print('RandomForestRegressor accuracy',RandForclass_err)
            print('DecisionTreeRegressor accuracy',DecTreeclass_err)
            # print('SVC accuracy',SVCclass_err)
            print('AdaRegressor accuracy',adaclass_err)
            print('LinearRegression  accuracy',LinReg_err)
            print('MLPRegressor accuracy',MLPClass_err)
            print('KNeighborsRegressor accuracy',KClass_err)
            print('XGBRegressor accuracy',XGB_err)
            print('KerasRegression accuracy ',keras_err)
            dict_models = {'Gradient': Gradclass_err,
                           'RandomForest': RandForclass_err,
                           'DecisionTree': DecTreeclass_err,
                           'Adaboost': adaclass_err,
                           'Lin': LinReg_err,
                           'Perceptron': MLPClass_err,
                           'Kneighbor': KClass_err,
                           'XGB': XGB_err,
                           'Keras': keras_err,
                           }
            model_name = max(dict_models, key=dict_models.get)
            print(f'Model with the highest accuracy: {model_name}')
            if model_name == 'Gradient':
                return Gradclass
            elif model_name == 'RandomForest':
                return RandForclass
            elif model_name == 'DecisionTree':
                return DecTreeclass
            elif model_name == 'Adaboost':
                return ada_class
            elif model_name == 'Lin':
                return LinReg
            elif model_name == 'Perceptron':
                return MLPClass
            elif model_name == 'Kneighbor':
                return KClass
            elif model_name == 'XGB':
                return xgb_class
            elif model_name == 'Keras':
                return model

    def predict_two_teams(self,model):
        while True:
            try:
                team_1 = input('team_1: ')
                if team_1 == 'exit':
                    break
                team_2 = input('team_2: ')
                year = input('year: ')
                team_1_url = 'https://www.sports-reference.com/cfb/schools/' + team_1.lower() + '/' + str(year) + '/gamelog/'
                team_2_url = 'https://www.sports-reference.com/cfb/schools/' + team_2.lower() + '/' + str(year) + '/gamelog/'
                team_1_df = html_to_df_web_scrape(team_1_url)
                team_2_df = html_to_df_web_scrape(team_2_url)
                #clean team 1 labels
                team_1_df['game_result'] = team_1_df['game_result'].str.replace('W','')
                team_1_df['game_result'] = team_1_df['game_result'].str.replace('L','')
                team_1_df['game_result'] = team_1_df['game_result'].str.replace('(','')
                team_1_df['game_result'] = team_1_df['game_result'].str.replace(')','')
                team_1_df['game_result'] = team_1_df['game_result'].str.split('-').str[0]
                team_1_df['game_result'] = team_1_df['game_result'].str.replace('-','')
                final_data_1 = team_1_df.replace(r'^\s*$', np.NaN, regex=True)
                #clean team 2 labels
                team_2_df['game_result'] = team_2_df['game_result'].str.replace('W','')
                team_2_df['game_result'] = team_2_df['game_result'].str.replace('L','')
                team_2_df['game_result'] = team_2_df['game_result'].str.replace('(','')
                team_2_df['game_result'] = team_2_df['game_result'].str.replace(')','')
                team_2_df['game_result'] = team_2_df['game_result'].str.split('-').str[0]
                team_2_df['game_result'] = team_2_df['game_result'].str.replace('-','')
                final_data_2 = team_2_df.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
                
                if 'Unnamed: 0' in final_data_1.columns:
                    final_data_1 = final_data_1.drop(columns=['Unnamed: 0'])
                if 'Unnamed: 0' in final_data_2.columns:
                    final_data_2 = final_data_2.drop(columns=['Unnamed: 0'])
                
                #drop cols
                final_data_1.drop(columns=self.drop_cols, inplace=True)
                final_data_2.drop(columns=self.drop_cols, inplace=True)
                final_data_1.drop(columns=['game_result'], inplace=True)
                final_data_2.drop(columns=['game_result'], inplace=True)
                
                #create data for prediction
                #TODO: create multiple features across different periods: all, last 2 games, 3 games, 4 games, 5 games
                df_features_1 = final_data_1.median(axis=0,skipna=True).to_frame().T
                df_features_2 = final_data_2.median(axis=0,skipna=True).to_frame().T
                
                #predict outcomes 
                if 'keras' in str(model):
                    score_val_1 = model.predict(df_features_1) #model.predict_classes?
                    score_val_2 = model.predict(df_features_2)
                    y_classes_1 = score_val_1.argmax(axis=-1) 
                    print(score_val_1)
                    print(y_classes_1)
                else:
                    score_val_1 = model.predict(df_features_1)
                    score_val_2 = model.predict(df_features_2)
                print(f'Score prediction for {team_1}: {score_val_1}')
                print(f'score prediction for {team_2}: {score_val_2}')
            except Exception as e:
                print(f'Team not found: {e}')

    def feature_importances(self,model):
        if model != "no model":
            if 'keras' in str(model):
                imps = PermutationImportance(model,random_state=1).fit(self.x_test, self.y_test)
                print(show_weights(imps,feature_names=self.x_test.columns))
            else:
                imps = permutation_importance(model, self.x_test, self.y_test)
            if 'MLPClassifier' or 'LinearRegression' or 'keras' in str(model):
                feature_imp = pd.Series(imps.importances_mean,index=self.x_test.columns).sort_values(ascending=False)
                plt.close()
                plt.figure()
                sns.barplot(x=feature_imp,y=feature_imp.index)
                plt.xlabel('Feature Importance')
                plt.ylabel('Features')
                title_name = f'FeatureImportance - {str(model)}'
                plt.title(title_name,fontdict={'fontsize': 6})
                save_name = 'FeatureImportance' + '.png'
                plt.tight_layout()
                plt.savefig(join(getcwd(), save_name), dpi=300)
            else:
                feature_imp = pd.Series(model.feature_importances_,index=self.x_test.columns).sort_values(ascending=False)
                plt.close()
                plt.figure()
                sns.barplot(x=feature_imp,y=feature_imp.index)
                plt.xlabel('Feature Importance')
                plt.ylabel('Features')
                title_name = f'FeatureImportance - {str(model)}'
                plt.title(title_name,fontdict={'fontsize': 6})
                save_name = 'FeatureImportanceRegress' + '.png'
                plt.tight_layout()
                plt.savefig(join(getcwd(), save_name), dpi=300)
def main():
    start_time = time.time()
    cfb_class = cfb_regressor()
    cfb_class.read_hyper_params()
    cfb_class.get_teams()
    cfb_class.split()
    model = cfb_class.machine()
    cfb_class.predict_two_teams(model)
    cfb_class.feature_importances(model)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()