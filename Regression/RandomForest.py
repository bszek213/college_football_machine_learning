#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Just execute the random forest regressor
@author: brianszekely
"""
from html_parse_cfb import html_to_df_web_scrape
# import argparse
from sportsipy.ncaaf.teams import Teams
from sportsipy.ncaaf.rankings import Rankings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.svm import SVR
# from sklearn.ensemble import AdaBoostRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Perceptron
# from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error #explained_variance_score
import time
from sklearn.model_selection import GridSearchCV
# from scipy.stats import uniform
from os import getcwd, mkdir
from os.path import join, exists
from scipy import stats
# from keras.utils import np_utils
# for modeling
# from keras.models import Sequential
# from keras.layers import Dense#, Dropout
# from keras.callbacks import EarlyStopping
import yaml
# from tensorflow.keras.metrics import RootMeanSquaredError
# import tensorflow as tf
# import xgboost as xgb
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
# import pickle
from tqdm import tqdm
# from sklearn.linear_model import PassiveAggressiveRegressor
import sys
# from sklearn import tree
# from subprocess import call
from time import sleep
from boruta import BorutaPy
#TODO: Build the keras hyperparam tuner
# Save models with pickle to avoid refitting time
import warnings
warnings.filterwarnings("ignore")
"""
NOTE: I am minMaxScaling the data, but I do not think 
with a RandomForest this would matter, as the node will
split on whatever float value of the feature that the greedy algorithm
at that level chose.
"""
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
        year_list_find = []
        year_list = [2022,2021,2019,2018,2017,2016,2015]#,2014,2013,,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
        if exists(join(getcwd(),'year_count.yaml')):
            with open(join(getcwd(),'year_count.yaml')) as file:
                year_counts = yaml.load(file, Loader=yaml.FullLoader)
        else:
            year_counts = {'year':year_list_find}
        if year_counts['year']:
            year_list_check =  year_counts['year']
            year_list_find = year_counts['year']
            year_list = [i for i in year_list if i not in year_list_check]
            print(f'Need data for year: {year_list}')
        if year_list:
            for year in year_list:
                all_teams = Teams(year)
                team_names = all_teams.dataframes.abbreviation
                team_names = team_names.sort_values()   
                final_list = []
                self.year_store = year
                for abv in tqdm(team_names):    
                    print(f'current team: {abv}, year: {year}')
                    # team = all_teams(abv)
                    str_combine = 'https://www.sports-reference.com/cfb/schools/' + abv.lower() + '/' + str(self.year_store) + '/gamelog/'
                    df_inst = html_to_df_web_scrape(str_combine,abv.lower(),self.year_store)
                    # df_inst = html_to_df_web_scrape(str_combine)
                    final_list.append(df_inst)
                output = pd.concat(final_list)
                output['game_result'] = output['game_result'].str.replace('W','')
                output['game_result'] = output['game_result'].str.replace('L','')
                output['game_result'] = output['game_result'].str.replace('(','')
                output['game_result'] = output['game_result'].str.replace(')','')
                output['game_result'] = output['game_result'].str.split('-').str[0]
                output['game_result'] = output['game_result'].str.replace('-','')
                final_data = output.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
                if exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data = pd.read_csv(join(getcwd(),'all_data_regressor.csv'))  
                self.all_data = pd.concat([self.all_data, final_data.dropna()])
                if not exists(join(getcwd(),'all_data_regressor.csv')):
                    self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'))
                self.all_data.to_csv(join(getcwd(),'all_data_regressor.csv'))
                year_list_find.append(year)
                print(f'year list after loop: {year_list_find}')
                with open(join(getcwd(),'year_count.yaml'), 'w') as write_file:
                    yaml.dump(year_counts, write_file)
                    print(f'writing {year} to yaml file')
        else:
            self.all_data = pd.read_csv(join(getcwd(),'all_data_regressor.csv'))
            print(f'size before removal of duplicates : {len(self.all_data)}')
            
        # else:
        #     self.all_data = pd.read_csv(final_dir)

    def split(self):
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.all_data = self.all_data.drop_duplicates(keep='first') #delete duplicate data
        print('len data after removal of duplicates: ', len(self.all_data))
        self.y = self.all_data['game_result']
        self.x = self.all_data.drop(columns=['game_result'])
        self.pre_process()

    def pre_process(self):        
        # Find features with correlation greater than 0.90
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.95)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        cols = self.x_no_corr.columns
        print(f'Columns dropped: {self.drop_cols}')
        #Remove outliers with 1.5 +/- IQR
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 25)
            Q3 = np.percentile(self.x_no_corr[col_name], 75)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+6.0*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-6.0*IQR)) 
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
        
        #minMaxScale
        self.minMax = MinMaxScaler()
        self.minMax.fit(self.x_no_corr)
        data_transfom = self.minMax.transform(self.x_no_corr)
        self.x_no_corr = pd.DataFrame(data_transfom,columns=self.x_no_corr.columns)
        #split data into train and test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_no_corr, self.y, train_size=0.8)
        cols = self.x_train.columns.to_list()
        self.x_train_cols = self.x_train.columns.to_list()
        self.y_train_cols = self.y_train.name
        for col_name in cols:
            # self.x_train[col_name], _ = stats.boxcox(self.x_train[col_name])
            self.prob_plots(col_name)
        #plot heat map
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
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
        if sys.argv[1] == 'tune':
            RandForclass = RandomForestRegressor()
            Rand_perm = {
                'criterion' : ["squared_error", "absolute_error", "poisson"],
                'n_estimators': range(100,500,51),
                'min_samples_split': np.arange(2, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            #['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted', 'matthews_corrcoef', 'max_error', 'mutual_info_score', 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'rand_score', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
            clf_rand = GridSearchCV(RandForclass, Rand_perm, scoring=['neg_root_mean_squared_error','explained_variance'],
                               refit='neg_root_mean_squared_error',verbose=4, n_jobs=-1)
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            print('RandomForestRegressor - best params: ',search_rand.best_params_)
            return 'no model'
        else:
            print('fit to model that has been tuned')
            RandForclass = RandomForestRegressor(criterion='squared_error',
                                                 bootstrap=True,
                                                 max_features='sqrt', 
                                                 min_samples_split=2, 
                                                 n_estimators=406
                                                 )#.fit(self.x_train,self.y_train)
            feat_selector = BorutaPy(
                verbose=2,
                estimator=RandForclass,
                n_estimators='auto',
                max_iter=10  # number of iterations to perform
            )
            feat_selector.fit(np.array(self.x_train),np.array(self.y_train))
            print(feat_selector.support_)
            print("\n------Support and Ranking for each feature------")
            self.drop_cols_boruta = []
            for i in range(len(feat_selector.support_)):
                if feat_selector.support_[i]:
                    print(f'Save feature: {self.x_train.columns[i]}')
                else:
                    print(f'Drop feature: {self.x_train.columns[i]}')
                    self.drop_cols_boruta.append(self.x_train.columns[i])
            print(f'Features to drop based on Boruta algorithm: {self.drop_cols_boruta}')
            self.x_train.drop(columns=self.drop_cols_boruta, inplace=True)
            self.x_test.drop(columns=self.drop_cols_boruta, inplace=True)
            RandForclass.fit(self.x_train,self.y_train)
            # RandForclass = RandomForestRegressor(criterion='absolute_error',
            #                                      bootstrap=True,
            #                                      max_features='sqrt', 
            #                                      min_samples_split=3, 
            #                                      n_estimators=400
            #                                      ).fit(self.x_train,self.y_train)
            RandForclass_err = r2_score(self.y_test, RandForclass.predict(self.x_test))
            print('RandomForestRegressor accuracy',RandForclass_err)
            RandForclass_err = np.sqrt(mean_squared_error(self.y_test, RandForclass.predict(self.x_test)))
            print('RandomForestRegressor rmse',RandForclass_err)
            return RandForclass
    def run_ma_predictions(self,data1,data2,team_1_loc,team_2_loc,model,team_1,team_2):
        """
        so far after running MA sub-analysis the best performing MA
        values are as follows: 2,4,8,14,16
        """
        list_ma = [2,4,8,14,16]
        list_outcomes = []
        count_1 = 0
        count_2 = 0
        if 'game_loc' not in self.drop_cols_boruta:
            data1['game_loc'] = team_1_loc
            data2['game_loc'] = team_2_loc
        for ma in list_ma:
            data1_ma = data1.dropna().rolling(ma).mean()
            data2_ma = data2.dropna().rolling(ma).mean()
            data1_ma = data1_ma.iloc[-1:]
            data2_ma = data2_ma.iloc[-1:]
            try:
                if model.predict(data1_ma) > model.predict(data2_ma):
                    list_outcomes.append(team_1)
                    count_1 += 1
                else:
                    list_outcomes.append(team_2)
                    count_2 += 1
            except:
                print(f'model.prediction returned NaN for {ma} ma value')
        print(f'Running average vote across 2,4,8,14,16 game intervals: {list_outcomes}')
        print(f'{team_1} win count: {count_1}')
        print(f'{team_2} win count: {count_2}')
        
    def predict_two_teams(self,model):
        while True:
            try:
                team_1 = input('team_1: ')
                if team_1 == 'exit':
                    break
                team_2 = input('team_2: ')
                print(f'is {team_1} home or away:')
                team_1_loc = input('type home or away: ')
                if team_1_loc == 'home':
                    team_2_loc = 0
                    team_1_loc = 1
                elif team_1_loc == 'away':
                    team_2_loc = 1
                    team_1_loc = 0
                # year = int(input('year: '))
                year = 2021
                #2021
                team_1_url = 'https://www.sports-reference.com/cfb/schools/' + team_1.lower() + '/' + str(year) + '/gamelog/'
                team_2_url = 'https://www.sports-reference.com/cfb/schools/' + team_2.lower() + '/' + str(year) + '/gamelog/'
                team_1_df2021 = html_to_df_web_scrape(team_1_url,team_1,year)
                team_2_df2021 = html_to_df_web_scrape(team_2_url,team_2,year)
                #2022
                year = 2022
                team_1_url = 'https://www.sports-reference.com/cfb/schools/' + team_1.lower() + '/' + str(year) + '/gamelog/'
                team_2_url = 'https://www.sports-reference.com/cfb/schools/' + team_2.lower() + '/' + str(year) + '/gamelog/'
                team_1_df2022= html_to_df_web_scrape(team_1_url,team_1,year)
                team_2_df2022 = html_to_df_web_scrape(team_2_url,team_2,year)
                #concatenate 2021 and 2022
                team_1_df = pd.concat([team_1_df2021, team_1_df2022])
                team_2_df = pd.concat([team_2_df2021, team_2_df2022])
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

                for col in final_data_1.columns:
                    if 'Unnamed' in col:
                        final_data_1.drop(columns=col,inplace=True)
                for col in final_data_2.columns:
                    if 'Unnamed' in col:
                        final_data_2.drop(columns=col,inplace=True)
                # if 'Unnamed: 0' in final_data_1.columns:
                #     final_data_1 = final_data_1.drop(columns=['Unnamed: 0'])
                # if 'Unnamed: 0' in final_data_2.columns:
                #     final_data_2 = final_data_2.drop(columns=['Unnamed: 0'])
                
                #drop cols
                final_data_1.drop(columns=self.drop_cols, inplace=True)
                final_data_2.drop(columns=self.drop_cols, inplace=True)
                final_data_1.drop(columns=['game_result'], inplace=True)
                final_data_2.drop(columns=['game_result'], inplace=True)
                #dropnans
                final_data_1.dropna(inplace=True)
                final_data_2.dropna(inplace=True)
                #Transform data with minMaxScaler
                transform_1 = self.minMax.transform(final_data_1)
                transform_2 = self.minMax.transform(final_data_2)
                final_data_1 = pd.DataFrame(transform_1,columns=final_data_1.columns)
                final_data_2 = pd.DataFrame(transform_2,columns=final_data_2.columns)
                final_data_1.drop(columns=self.drop_cols_boruta, inplace=True)
                final_data_2.drop(columns=self.drop_cols_boruta, inplace=True)
                #create data for prediction
                df_features_1 = final_data_1.dropna().median(axis=0,skipna=True).to_frame().T
                df_features_2 = final_data_2.dropna().median(axis=0,skipna=True).to_frame().T
                team_1_total = 0
                team_2_total = 0
                # #calculate running average short and long intervals
                # data1_long = final_data_1.dropna().rolling(6).mean() #long
                # data2_long = final_data_2.dropna().rolling(6).mean()
                # data1_long = data1_long.iloc[-1:]
                # data2_long = data2_long.iloc[-1:]
                # data1_short = final_data_1.dropna().rolling(2).mean() #long
                # data2_short= final_data_2.dropna().rolling(2).mean()
                # data1_short = data1_short.iloc[-1:]
                # data2_short = data2_short.iloc[-1:]
                # data1_med = final_data_1.dropna().rolling(4).mean() #medium
                # data2_med= final_data_2.dropna().rolling(4).mean()
                # data1_med = data1_med.iloc[-1:]
                # data2_med = data2_med.iloc[-1:]
                # if not data1_long.isnull().values.any() and not data1_short.isnull().values.any():
                #     if 'game_loc' not in self.drop_cols_boruta:
                #         data1_long['game_loc'] = team_1_loc
                #         data2_long['game_loc'] = team_2_loc
                #         data1_short['game_loc'] = team_1_loc
                #         data2_short['game_loc'] = team_2_loc
                #         data1_med['game_loc'] = team_1_loc
                #         data2_med['game_loc'] = team_2_loc
                #     team_1_data_long_avg = model.predict(data1_long)
                #     team_2_data_long_avg = model.predict(data2_long)
                #     team_1_data_short_avg = model.predict(data1_short)
                #     team_2_data_short_avg = model.predict(data2_short)
                #     team_1_data_med_avg = model.predict(data1_med)
                #     team_2_data_med_avg = model.predict(data2_med)
                #ROLLING AVG CHECK TO SEE IF I CAN GET A MA VALUE THAT PERFECTLY PREDICTS WHAT THE SCORE WAS
                
                # range_ma = np.arange(2,len(final_data_1),1)
                # team_1_actual = int(input(f'{team_1} score: '))
                # team_2_actual = int(input(f'{team_2} score: '))
                # team_1_dict = {}
                # team_2_dict = {}
                # for val in range_ma:
                #     data1_check = final_data_1.dropna().rolling(val).mean() #medium
                #     data2_check= final_data_2.dropna().rolling(val).mean()
                #     data1_check = data1_check.iloc[-1:]
                #     data2_check = data2_check.iloc[-1:]
                #     if 'game_loc' not in self.drop_cols_boruta:
                #         data1_check['game_loc'] = team_1_loc
                #         data2_check['game_loc'] = team_2_loc
                #     try:
                #         score_1 = model.predict(data1_check)
                #         score_2 = model.predict(data2_check)
                #         team_1_dict[val] = abs(score_1[0]-team_1_actual)
                #         team_2_dict[val] = abs(score_2[0]-team_2_actual)
                #     except:
                #         print(f'Prediction failed for MA value {val}')
                # file1 = open("finding_best_ma.txt", "a")
                # combine1 = str(min(team_1_dict.items(), key=lambda x: x[1])) + "\n"
                # combine2 = str(min(team_2_dict.items(), key=lambda x: x[1])) + "\n"
                # file1.write(combine1)
                # file1.write(combine2)
                # file1.close()
                print('============================================================')
                data1 = final_data_1.dropna().median(axis=0,skipna=True).to_frame().T
                data2 = final_data_2.dropna().median(axis=0,skipna=True).to_frame().T
                if 'game_loc' not in self.drop_cols_boruta:
                    data1['game_loc'] = team_1_loc
                    data2['game_loc'] = team_2_loc
                game_won_team_1 = []
                game_won_team_2 = []
                if not data1.isnull().values.any() and not data1.isnull().values.any():
                    team_1_data_all = model.predict(data1)
                    team_2_data_all = model.predict(data2)
                    if team_1_data_all[0] > team_2_data_all[0]:
                        team_1_total += 1
                        game_won_team_1.append('season')
                    else:
                        team_2_total += 1
                        game_won_team_2.append('season')
                    print(f'Score prediction for {team_1} across 2021 and 2022 season: {team_1_data_all[0]} points')
                    print(f'Score prediction for {team_2} across 2021 and 2022 season: {team_2_data_all[0]} points')
                    print('====')
                data1 = final_data_1.iloc[-1:].dropna().median(axis=0,skipna=True).to_frame().T
                data2 = final_data_2.iloc[-1:].dropna().median(axis=0,skipna=True).to_frame().T
                if not data1.isnull().values.any() and not data1.isnull().values.any():
                    team_1_data_last = model.predict(data1)
                    team_2_data_last = model.predict(data2)
                    if team_1_data_last[0] > team_2_data_last[0]:
                        team_1_total += 1
                        game_won_team_1.append('last_game')
                    else:
                        team_2_total += 1
                        game_won_team_2.append('last_game')
                    print(f'Score prediction for {team_1} last game: {team_1_data_last[0]} points')
                    print(f'Score prediction for {team_2} last game: {team_2_data_last[0]} points')
                    print('====')
                data1 = final_data_1.iloc[-2:].dropna().median(axis=0,skipna=True).to_frame().T
                data2 = final_data_2.iloc[-2:].dropna().median(axis=0,skipna=True).to_frame().T
                if not data1.isnull().values.any() and not data1.isnull().values.any():
                    team_1_data_last2 = model.predict(data1)
                    team_2_data_last2 = model.predict(data2)
                    if team_1_data_last2[0] > team_2_data_last2[0]:
                        team_1_total += 1
                        game_won_team_1.append('last_2_games')
                    else:
                        team_2_total += 1
                        game_won_team_2.append('last_2_games')
                    print(f'Score prediction for {team_1} last 2 game: {team_1_data_last2[0]} points')
                    print(f'Score prediction for {team_2} last 2 game: {team_2_data_last2[0]} points')
                    print('====')
                data1 = final_data_1.iloc[-3:].dropna().median(axis=0,skipna=True).to_frame().T
                data2 = final_data_2.iloc[-3:].dropna().median(axis=0,skipna=True).to_frame().T
                if not data1.isnull().values.any() and not data1.isnull().values.any():
                    team_1_data_last3 = model.predict(data1)
                    team_2_data_last3 = model.predict(data2)
                    if team_1_data_last3[0] > team_2_data_last3[0]:
                        team_1_total += 1
                        game_won_team_1.append('last_3_games')
                    else:
                        team_2_total += 1
                        game_won_team_2.append('last_3_games')
                    print(f'Score prediction for {team_1} last 3 game: {team_1_data_last3[0]} points')
                    print(f'Score prediction for {team_2} last 3 game: {team_2_data_last3[0]} points')
                    print('====')
                data1 = final_data_1.iloc[-5:].dropna().median(axis=0,skipna=True).to_frame().T
                data2 = final_data_2.iloc[-5:].dropna().median(axis=0,skipna=True).to_frame().T
                if not data1.isnull().values.any() and not data1.isnull().values.any():
                    team_1_data_last5 = model.predict(data1)
                    team_2_data_last5 = model.predict(data2)
                    if team_1_data_last5[0] > team_2_data_last5[0]:
                        team_1_total += 1
                        game_won_team_1.append('last_5_games')
                    else:
                        team_2_total += 1
                        game_won_team_2.append('last_5_games')
                    print(f'Score prediction for {team_1} last 5 game: {team_1_data_last5[0]} points')
                    print(f'Score prediction for {team_2} last 5 game: {team_2_data_last5[0]} points')
                # print('===============================================================')
                # print(f'Score prediction for {team_1} running average long: {team_1_data_long_avg[0]} points')
                # print(f'Score prediction for {team_2} running average long: {team_2_data_long_avg[0]} points')
                # print(f'Score prediction for {team_1} running average short: {team_1_data_short_avg[0]} points')
                # print(f'Score prediction for {team_2} running average short: {team_2_data_short_avg[0]} points')
                # print(f'Score prediction for {team_1} running average medium: {team_1_data_med_avg[0]} points')
                # print(f'Score prediction for {team_2} running average medium: {team_2_data_med_avg[0]} points')
                print('===============================================================')
                # vote_running_avg = []
                # if team_1_data_long_avg[0] > team_2_data_long_avg[0]:
                #     vote_running_avg.append(team_1)
                # else:
                #     vote_running_avg.append(team_2)
                # if team_1_data_short_avg[0] > team_2_data_short_avg[0]:
                #     vote_running_avg.append(team_1)
                # else:
                #     vote_running_avg.append(team_2)
                # if team_1_data_med_avg[0] > team_2_data_med_avg[0]:
                #     vote_running_avg.append(team_1)
                # else:
                #     vote_running_avg.append(team_2)
                print('====Matchup win count=====')
                # print(f'Running average vote across short, medium, and long intervals: {vote_running_avg}')
                print(f'{team_1} total: {team_1_total} : games won: {game_won_team_1}')
                print(f'{team_2} total: {team_2_total} : games won: {game_won_team_2}')
                print('====Running average analysis=====')
                self.run_ma_predictions(final_data_1,final_data_2,team_1_loc,team_2_loc,model,team_1,team_2)
                print('===============================================================')
                # score_val_1 = model.predict(df_features_1)
                # score_val_2 = model.predict(df_features_2)
                #predict outcomes 
                if 'keras' in str(model):
                    score_val_1 = model.predict(df_features_1) #model.predict_classes?
                    score_val_2 = model.predict(df_features_2)
                    y_classes_1 = score_val_1.argmax(axis=-1) 
                    print(score_val_1)
                    print(y_classes_1)
                    print(f'Score prediction for {team_1}: {score_val_1}')
                    print(f'score prediction for {team_2}: {score_val_2}')
                    print('====')
                # else:
                #     score_val_1 = model.predict(df_features_1)
                #     score_val_2 = model.predict(df_features_2)
                # print(f'Score prediction for {team_1}: {score_val_1}')
                # print(f'score prediction for {team_2}: {score_val_2}')
            except Exception as e:
                print(f'Team not found: {e}')

    def feature_importances(self,model):
        if model != "no model":
            if 'keras' in str(model):
                imps = PermutationImportance(model,random_state=1).fit(self.x_test, self.y_test)
                print(show_weights(imps,feature_names=self.x_test.columns))
            else:
                imps = permutation_importance(model, self.x_test, self.y_test)
            if 'MLPClassifier' or 'LinearRegression' or 'PassiveAggressive' or 'keras' in str(model):
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
    if not sys.argv[1] == 'tune':
        cfb_class.predict_two_teams(model)
        cfb_class.feature_importances(model)
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main()