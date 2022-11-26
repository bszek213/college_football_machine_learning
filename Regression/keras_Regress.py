from html_parse_cfb import html_to_df_web_scrape
# import argparse
from sportsipy.ncaaf.teams import Teams
from sportsipy.ncaaf.rankings import Rankings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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
from keras.models import Sequential
from keras.layers import Dense#, Dropout
# from keras.callbacks import EarlyStopping
import yaml
from tensorflow.keras.metrics import RootMeanSquaredError
import tensorflow as tf
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
# import pickle
from tqdm import tqdm
# from sklearn.linear_model import PassiveAggressiveRegressor
import sys
class kerasRegressor():
    def __init__(self):
        print('Default Constructor kerasRegressor')
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
        print('len data: ', len(self.all_data))
            
        # else:
        #     self.all_data = pd.read_csv(final_dir)

    def split(self):
        for col in self.all_data.columns:
            if 'Unnamed' in col:
                self.all_data.drop(columns=col,inplace=True)
        self.y = self.all_data['game_result']
        self.x = self.all_data.drop(columns=['game_result'])
        self.pre_process()
    def pre_process(self):        
        # Find features with correlation greater than 0.90
        corr_matrix = np.abs(self.x.astype(float).corr())
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.8)]
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
            upper = np.where(self.x_no_corr[col_name] >= (Q3+1.5*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-1.5*IQR)) 
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

    def kerasMachine(self):
        #Keras classifier 
            model = Sequential()
            # model.add(LSTM(12))
            # scaler = StandardScaler()
            # # scaler = MinMaxScaler()(feature_range=(0, 1))
            # scaled_data = scaler.fit_transform(self.x_train)
            # scaled_train = pd.DataFrame(scaled_data, columns = self.x_train.columns)
            # scaled_data_test = scaler.fit_transform(self.x_test)
            # scaled_test = pd.DataFrame(scaled_data_test, columns = self.x_test.columns)
            model.add(Dense(8, input_shape=(self.x_train.shape[1],), activation="linear"))#input shape - (features,)
            # model.add(Dropout(0.3))
            model.add(Dense(6, activation='relu'))
            model.add(Dense(4, activation='linear'))
            model.add(Dense(2, activation='linear'))
            model.add(Dense(1, activation='softmax'))
            model.summary() 
            #compile 
            model.compile(optimizer='adam', 
                  loss='mse',
                  metrics=[RootMeanSquaredError()])
            history = model.fit(self.x_train,
                        self.y_train,
                        # callbacks=[es],
                        epochs=500,# you can set this to a big number!
                        batch_size=20,          
                        validation_data=(self.x_test, self.y_test),
                        shuffle=True,
                        workers=8, #change this to be the num cores
                        verbose=1)
            # keras_acc = history.history['accuracy']
            # pred_train = history.predict(self.x_test) #will need this in the future when I want to look at one team vs. another
            # scaled_data = scaler.fit_transform(self.x_test)
            scaled_test = pd.DataFrame(self.x_test, columns = self.x_test.columns)
            scores = model.evaluate(scaled_test, self.y_test)
            keras_y_predict = model.predict(scaled_test)
def main():
    start_time = time.time()
    cfb_class = kerasRegressor()
    cfb_class.read_hyper_params()
    cfb_class.get_teams()
    cfb_class.split()
    model = cfb_class.kerasMachine()
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main()