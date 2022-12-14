# -*- coding: utf-8 -*-
"""
College football game classifier
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
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
from sklearn.model_selection import GridSearchCV
# from scipy.stats import uniform
from os import getcwd
from os.path import join, exists
from scipy import stats
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import yaml
import tensorflow as tf
import xgboost as xgb
from sklearn.inspection import permutation_importance
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
# from tensorflow.keras.layers import LSTM
#TODO: 1. Add PCA to reduce unnecessary features
#      3. Try different optimizers, activations 
def keras_model(unit):
    #Keras classifier 
    model = Sequential()
    # model.add(LSTM(12))
    model.add(Dense(units=unit,input_shape=(shape_x_train,),activation="relu"))#input shape - (features,) = input_shape=(self.x_train.shape[1],),
    # model.add(Dropout(0.3))
    model.add(Dense(units=unit,activation='relu'))
    model.add(Dense(units=unit,activation='softmax'))
    model.add(Dense(units=unit,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary() 
    #compile 
    model.compile(optimizer='SGD', 
          loss='binary_crossentropy', #tf.keras.losses.BinaryCrossentropy(from_logits=True)
          metrics=['accuracy'])
    #stop training when the model has not improved after 10 steps
    # es = EarlyStopping(monitor='val_accuracy', 
    #                            mode='max', # don't minimize the accuracy!
    #                            patience=20,
    #                            restore_best_weights=True)
    # history = model.fit(self.x_train,
    #             self.y_train,
    #             # callbacks=[es],
    #             epochs=500, # you can set this to a big number!
    #             batch_size=20,
    #             # validation_data=(self.x_test, self.y_test),
    #             shuffle=True,
    #             verbose=1)
    return model
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
    def read_hyper_params(self):
        final_dir = join(getcwd(), 'hyper_params.yaml')
        isExists = exists(final_dir)
        if isExists == True:
            with open(final_dir) as file:
                self.hyper_param_dict = yaml.load(file, Loader=yaml.FullLoader)

    def get_teams(self):
        final_dir = join(getcwd(), 'all_data.csv')
        isExists = exists(final_dir)
        if isExists == False:
            year_list = [2021,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
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

        #Create new scaled data - DO I REMOVE THE VARIABLES THAT ARE HIGHLY 
        # CORRELATED BEFORE I STANDARDIZE THEM OR STANDARDIZE AND THEN REMOVE
        # HIGHLY CORRELATED
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_data = scaler.fit_transform(self.x_train)
        # scaler = StandardScaler()
        # scaled_data = scaler.fit(self.x_train).transform(self.x_train)
        # self.x_train = pd.DataFrame(scaled_data, columns = cols)
        #Probability plots after or before scaling the data? Right now I am 
        # doing it after scaling
        # Peform a box cox transform
        # use z-score?
        # stats.zscore(x, axis=1, nan_policy='omit')
        for col_name in cols:
            # self.x_train[col_name], _ = stats.boxcox(self.x_train[col_name])
            self.prob_plots(col_name)
        #plot heat map
        top_corr_features = corr_matrix.index
        plt.figure(figsize=(20,20))
        g=sns.heatmap(corr_matrix[top_corr_features],annot=True,cmap="RdYlGn")
        plt.savefig('correlations.png')
        plt.close()
        
    def machine(self):
        #Drop data that poorly fit the normally distribution
        # self.x_train.drop(columns=['turnovers','first_down_penalty','fumbles_lost','pass_int'], inplace=True)
        # self.x_test.drop(columns=['turnovers','first_down_penalty','fumbles_lost','pass_int'], inplace=True)
        #load in the hyperparams from file if the file exists
        final_dir = join(getcwd(), 'hyper_params.yaml')
        isExists = exists(final_dir)
        if isExists == True:
            Gradclass = GradientBoostingClassifier(**self.hyper_param_dict['GradientBoostingClassifier']).fit(self.x_train,self.y_train)
            RandForclass = RandomForestClassifier(**self.hyper_param_dict['RandomForestClassifier']).fit(self.x_train,self.y_train)
            ada_class = AdaBoostClassifier(**self.hyper_param_dict['AdaClassifier']).fit(self.x_train,self.y_train)
            DecTreeclass = DecisionTreeClassifier(**self.hyper_param_dict['DecisionTreeClassifier']).fit(self.x_train,self.y_train)
            LogReg = LogisticRegression(**self.hyper_param_dict['LogisticRegression']).fit(self.x_train,self.y_train)
            KClass = KNeighborsClassifier(**self.hyper_param_dict['KNeighborsClassifier']).fit(self.x_train,self.y_train)
            MLPClass = MLPClassifier(**self.hyper_param_dict['MLPClassifier']).fit(self.x_train,self.y_train)
            xgb_class = xgb.XGBClassifier(**self.hyper_param_dict['XGB-boost']).fit(self.x_train,self.y_train)  
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
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
            #stop training when the model has not improved after 10 steps
            es = EarlyStopping(monitor='val_accuracy', 
                                       mode='max', # don't minimize the accuracy!
                                       patience=20,
                                       restore_best_weights=True)
            # history = model.fit(self.x_train,
            #             self.y_train,
            #             # callbacks=[es],
            #             epochs=500, # you can set this to a big number!
            #             batch_size=20,
            #             # validation_data=(self.x_test, self.y_test),
            #             shuffle=True,
            #             verbose=1)
            history = model.fit(self.x_train,
                        self.y_train,
                        # callbacks=[es],
                        epochs=400, # you can set this to a big number!
                        batch_size=20,
                        validation_split=0.2,           
                        # validation_data=(self.x_test, self.y_test),
                        shuffle=True,
                        verbose=1)
            # keras_acc = history.history['accuracy']
            # pred_train = history.predict(self.x_test) #will need this in the future when I want to look at one team vs. another
            scores = model.evaluate(self.x_test, self.y_test, verbose=0)
            plt.figure()
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.ylim(0, 1)
            plt.title('Keras Classifier Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(['train','test'])
            save_name = 'keras_model_acc' + '.png'
            plt.savefig(join(getcwd(),save_name), dpi=200)
            plt.close()
        else:
            Gradclass = GradientBoostingClassifier()
            Grad_perm = {
                'loss' : ['log_loss', 'exponential'],
                'learning_rate': np.arange(0.1, .5, 0.1, dtype=float),
                'n_estimators': range(100,500,100),
                'criterion' : ['friedman_mse', 'squared_error'],
                'max_depth': np.arange(1, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            clf = GridSearchCV(Gradclass, Grad_perm, scoring=['accuracy'],
                                refit='accuracy', verbose=4, n_jobs=-1) #cv=5
            # param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
            # clf = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, 
            #                                                           n_estimators=60, 
            #                                                           max_features='sqrt', 
            #                                                           subsample=0.8, 
            #                                                           random_state=10), 
            #                         param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
            search_Grad = clf.fit(self.x_train,self.y_train)
            
            RandForclass = RandomForestClassifier()
            Rand_perm = {
                'criterion' : ["gini", "entropy"],
                'n_estimators': range(100,500,100),
                'max_depth': np.arange(1, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            clf_rand = GridSearchCV(RandForclass, Rand_perm, scoring=['accuracy'],
                               refit='accuracy',verbose=4, n_jobs=-1)
            search_rand = clf_rand.fit(self.x_train,self.y_train)
            # RandForclass.fit(self.x_train,self.y_train)
            
            DecTreeclass = DecisionTreeClassifier()
            Dec_perm = {
                'splitter' : ["best", "random"],
                'criterion' : ["gini", "entropy"],
                'max_depth': np.arange(1, 5, 1, dtype=int),
                'max_features' : [1, 'sqrt', 'log2']
                }
            clf_dec = GridSearchCV(DecTreeclass, Dec_perm, scoring=['accuracy'],
                               refit='accuracy',verbose=4, n_jobs=-1)
            search_dec = clf_dec.fit(self.x_train,self.y_train)

            ada_class = AdaBoostClassifier()
            ada_perm = {'n_estimators': range(50,200,50),
                          'learning_rate': np.arange(.5,2.5,.5,dtype=float),
                          'algorithm': ['SAMME', 'SAMME.R']}
            clf_ada = GridSearchCV(ada_class, ada_perm, scoring=['accuracy'],
                                refit='accuracy', verbose=4, n_jobs=-1)
            search_ada = clf_ada.fit(self.x_train,self.y_train)
            LogReg = LogisticRegression()
            log_reg_perm = {
                'penalty': ['l2'],
                'C': np.arange(1, 5, 0.5, dtype=float),
                'max_iter': range(100,1000,100),
                'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
                }
            clf_Log = GridSearchCV(LogReg, log_reg_perm, scoring=['accuracy'],
                               refit='accuracy', verbose=4, n_jobs=-1)
            search_Log = clf_Log.fit(self.x_train,self.y_train)
            
            MLPClass = MLPClassifier()
            MLP_perm = {
                'solver' : ['lbfgs', 'sgd', 'adam'],
                'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init' : np.arange(0.001, 0.005, 0.001, dtype=float),
                'max_iter': range(100,1000,200),
                # 'tol': np.arange(0.001, 0.005, 0.001, dtype=float)
                }
            clf_MLP = GridSearchCV(MLPClass, MLP_perm, scoring=['accuracy'],
                               refit='accuracy', verbose=4, n_jobs=-1)
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
                               refit='accuracy', verbose=4, n_jobs=-1)
            search_KClass= clf_KClass.fit(self.x_train,self.y_train)
            #Keras tuning
            global shape_x_train
            shape_x_train = self.x_train.shape[1]
            model_keras = KerasClassifier(build_fn=keras_model)
            params_keras = {'batch_size': [100,50,32,25,20],
                            'nb_epoch': [100,200,300,400],
                            'unit':[4,8,12,16,32]}
            keras_grid = GridSearchCV(estimator=model_keras, param_grid=params_keras,cv=5,n_jobs=-1)
            keras_grid.fit(self.x_train,self.y_train)
            
            #XGB boost tuning
            estimator = xgb.XGBClassifier(
                    objective= 'binary:logistic',
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
                                        scoring = 'accuracy',
                                        n_jobs = -1,
                                        cv = 5,
                                        verbose=4
                                        )
    
            grid_search_xgb.fit(self.x_train,self.y_train)
            
        # SVCclass = SVC()
        # SVC_perm = {'C': [0.1,1, 10, 100],
        #               'gamma': [1,0.1,0.01,0.001],
        #               'kernel': ['rbf', 'poly', 'sigmoid']}
        # clf_SVC = GridSearchCV(SVCclass, SVC_perm, scoring=['accuracy'],
        #                    refit='accuracy', verbose=4, n_jobs=1)
        # search_SVC = clf_SVC.fit(self.x_train,self.y_train) #This model for some reason freezes here on SVC

        # PerClass_err = accuracy_score(self.y_test, PerClass.predict(self.x_test))
        print('Removed features (>=0.75 correlation): ', self.drop_cols)
        if isExists == False:
            print('GradientBoostingClassifier - best params: ',search_Grad.best_params_)
            print('RandomForestClassifier - best params: ',search_rand.best_params_)
            print('DecisionTreeClassifier - best params: ',search_dec.best_params_)
            # print('SVC - best params: ',search_SVC.best_params_)
            print('AdaClassifier - best params: ',search_ada.best_params_)
            print('LogisticRegression - best params:',search_Log.best_params_)
            print('MLPClassifier - best params: ',search_MLP.best_params_)
            print('KNeighborsClassifier - best params: ',search_KClass.best_params_)
            print('XGB-boost - best params: ',grid_search_xgb.best_params_)
            Gradclass_err = accuracy_score(self.y_test, search_Grad.predict(self.x_test))
            RandForclass_err = accuracy_score(self.y_test, search_rand.predict(self.x_test))
            DecTreeclass_err = accuracy_score(self.y_test, search_dec.predict(self.x_test))
            # SVCclass_err = accuracy_score(self.y_test, search_SVC.predict(self.x_test))
            adaclass_err = accuracy_score(self.y_test, search_ada.predict(self.x_test))
            LogReg_err = accuracy_score(self.y_test, search_Log.predict(self.x_test))
            MLPClass_err = accuracy_score(self.y_test, search_MLP.predict(self.x_test))
            KClass_err = accuracy_score(self.y_test, search_KClass.predict(self.x_test))
            XGB_err = accuracy_score(self.y_test, grid_search_xgb.predict(self.x_test))
            print(f'Keras best params: {keras_grid.best_score_}, {keras_grid.best_params_}')
            print('GradientBoostingClassifier accuracy',Gradclass_err)
            print('RandomForestClassifier accuracy',RandForclass_err)
            print('DecisionTreeClassifier accuracy',DecTreeclass_err)
            # print('SVC accuracy',SVCclass_err)
            print('AdaClassifier accuracy',adaclass_err)
            print('LogisticRegression  accuracy',LogReg_err)
            print('MLPClassifier accuracy',MLPClass_err)
            print('KNeighborsClassifier accuracy',KClass_err)
            print('XGBClassifier accuracy',XGB_err)
            return 'no model'
        else:
            Gradclass_err = accuracy_score(self.y_test, Gradclass.predict(self.x_test))
            RandForclass_err = accuracy_score(self.y_test, RandForclass.predict(self.x_test))
            DecTreeclass_err = accuracy_score(self.y_test, DecTreeclass.predict(self.x_test))
            adaclass_err = accuracy_score(self.y_test, ada_class.predict(self.x_test))
            LogReg_err = accuracy_score(self.y_test, LogReg.predict(self.x_test))
            MLPClass_err = accuracy_score(self.y_test, MLPClass.predict(self.x_test))
            KClass_err = accuracy_score(self.y_test, KClass.predict(self.x_test))
            xgb_class_err = accuracy_score(self.y_test, xgb_class.predict(self.x_test))
            print('GradientBoostingClassifier accuracy',Gradclass_err)
            print('RandomForestClassifier accuracy',RandForclass_err)
            print('DecisionTreeClassifier accuracy',DecTreeclass_err)
            # print('SVC accuracy',SVCclass_err)
            print('AdaClassifier accuracy',adaclass_err)
            print('LogisticRegression  accuracy',LogReg_err)
            print('MLPClassifier accuracy',MLPClass_err)
            print('KNeighborsClassifier accuracy',KClass_err)
            print('XGBClassifier accuracy',xgb_class_err)
            print("KerasClassifier: test loss, test acc:", scores)
            # print('KerasClassifier accuracy', np.mean(keras_acc))
            print('check the amount of wins and losses are in the training label data: ',self.y_train.value_counts())
            #return model with the highest accuracy
            dict_models = {'Gradient': Gradclass_err,
                           'RandomForest': RandForclass_err,
                           'DecisionTree': DecTreeclass_err,
                           'Adaboost': adaclass_err,
                           'Log': LogReg_err,
                           'Perceptron': MLPClass_err,
                           'Kneighbor': KClass_err,
                           'XGB': xgb_class_err,
                           'Keras': scores[1],
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
            elif model_name == 'Log':
                return LogReg
            elif model_name == 'Perceptron':
                return MLPClass
            elif model_name == 'Kneighbor':
                return KClass
            elif model_name == 'XGB':
                return xgb_class
            elif model_name == 'Keras':
                return model
    
    def prob_plots(self,col_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        prob = stats.probplot(self.x_train[col_name], dist=stats.norm, plot=ax1)
        title = f'probPlot of training data against normal distribution, feature: {col_name}'  
        ax1.set_title(title,fontsize=10)
        save_name = 'probplot_' + col_name + '.png'
        plt.tight_layout()
        plt.savefig(join(getcwd(), 'prob_plots',save_name), dpi=200)
        # plt.close()
        
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
                #team 1 create labels
                team_1_df['game_result'].loc[team_1_df['game_result'].str.contains('W')] = 'W'
                team_1_df['game_result'].loc[team_1_df['game_result'].str.contains('L')] = 'L'
                team_1_df['game_result'] = team_1_df['game_result'].replace({'W': 1, 'L': 0})
                final_data_1 = team_1_df.replace(r'^\s*$', np.NaN, regex=True) #replace empty string with NAN
                #team 2 create labels
                team_2_df['game_result'].loc[team_2_df['game_result'].str.contains('W')] = 'W'
                team_2_df['game_result'].loc[team_2_df['game_result'].str.contains('L')] = 'L'
                team_2_df['game_result'] = team_2_df['game_result'].replace({'W': 1, 'L': 0})
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
                
                #Subtraction method
                features1_np = df_features_1.to_numpy()
                features2_np = df_features_2.to_numpy()
                diff = [a-b for a,b in zip(features1_np,features2_np)]
                arr = np.array(diff)
                nx,ny = arr.shape
                final_vector = arr.reshape((1,nx*ny))
                cols = df_features_1.columns
                final_vector_1 = pd.DataFrame(final_vector, columns=cols)
                if 'keras' in str(model):
                    win_loss_1 = model.predict(final_vector_1) #model.predict_classes?
                    win_loss_2 = model.predict(df_features_2)
                    y_classes_1 = win_loss_1.argmax(axis=-1) 
                    print(win_loss_1)
                    print(y_classes_1)
                else:
                    Probability_win_loss_1 = model.predict_proba(df_features_1)
                    Probability_win_loss_2 = model.predict_proba(df_features_2)
                    win_loss_1 = model.predict(df_features_1)
                    win_loss_2 = model.predict(df_features_2)
                #     Probability_win_loss = model.predict_proba(final_vector_1)
                #     win_loss = model.predict(final_vector_1)
                print(f'Prediction for {team_1}: {win_loss_1}')
                print(f'Prediction for {team_2}: {win_loss_2}')
                print(f'{team_1} loss proba: {Probability_win_loss_1[0][0]},win proba: {Probability_win_loss_1[0][1]}')
                print(f'{team_2} loss proba: {Probability_win_loss_2[0][0]},win proba: {Probability_win_loss_2[0][1]}')
            except Exception as e:
                print(f'Team not found: {e}')
        
    def feature_importances(self,model):
        if model != "no model":
            if 'keras' in str(model):
                imps = PermutationImportance(model,random_state=1).fit(self.x_test, self.y_test)
                print(show_weights(imps,feature_names=self.x_test.columns))
            else:
                imps = permutation_importance(model, self.x_test, self.y_test)
            if 'MLPClassifier' or 'LogisticRegression' or 'keras' in str(model):
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
                save_name = 'FeatureImportance' + '.png'
                plt.tight_layout()
                plt.savefig(join(getcwd(), save_name), dpi=300)
        

def main():
    start_time = time.time()
    cfb_class = cfb()
    cfb_class.input_arg()
    cfb_class.read_hyper_params()
    cfb_class.get_teams()
    cfb_class.split()
    model = cfb_class.machine()
    cfb_class.predict_two_teams(model)
    cfb_class.feature_importances(model)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()