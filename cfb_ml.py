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
#TODO: 1. Add PCA to reduce unnecessary features
#      2. plot of the accuracy over epochs
#      3. Try different optimizers, activations 
def keras_model(unit):
    #Keras classifier 
    model = Sequential()
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
        #dict_keys(['GradientBoostingClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier',
        #'AdaClassifier', 'LogisticRegression', 'MLPClassifier', 'KNeighborsClassifier'])

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
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.90)]
        self.drop_cols = to_drop
        self.x_no_corr = self.x.drop(columns=to_drop)
        cols = self.x_no_corr.columns
        
        #Remove outliers with 1.5 +/- IQR
        print(f'old feature dataframe shape before outlier removal: {self.x_no_corr.shape}')
        for col_name in cols:
            Q1 = np.percentile(self.x_no_corr[col_name], 25)
            Q3 = np.percentile(self.x_no_corr[col_name], 75)
            IQR = Q3 - Q1
            upper = np.where(self.x_no_corr[col_name] >= (Q3+3.0*IQR)) #1.5 is the standard, use two to see if more data helps improve model performance
            lower = np.where(self.x_no_corr[col_name] <= (Q1-3.0*IQR)) 
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
            #Keras classifier 
            model = Sequential()
            model.add(Dense(8, input_shape=(self.x_train.shape[1],), activation="relu"))#input shape - (features,)
            # model.add(Dropout(0.3))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(8, activation='softmax'))
            model.add(Dense(8, activation='relu'))
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
                        epochs=1000, # you can set this to a big number!
                        batch_size=32,
                        validation_split=0.25,           
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
            keras_grid = GridSearchCV(estimator=model_keras, param_grid=params_keras,cv=5)
            keras_grid.fit(self.x_train,self.y_train)
        
        # SVCclass = SVC()
        # SVC_perm = {'C': [0.1,1, 10, 100],
        #               'gamma': [1,0.1,0.01,0.001],
        #               'kernel': ['rbf', 'poly', 'sigmoid']}
        # clf_SVC = GridSearchCV(SVCclass, SVC_perm, scoring=['accuracy'],
        #                    refit='accuracy', verbose=4, n_jobs=1)
        # search_SVC = clf_SVC.fit(self.x_train,self.y_train) #This model for some reason freezes here on SVC

        # PerClass_err = accuracy_score(self.y_test, PerClass.predict(self.x_test))
        print('Removed features (>=0.90 correlation): ', self.drop_cols)
        if isExists == False:
            print('GradientBoostingClassifier - best params: ',search_Grad.best_params_)
            print('RandomForestClassifier - best params: ',search_rand.best_params_)
            print('DecisionTreeClassifier - best params: ',search_dec.best_params_)
            # print('SVC - best params: ',search_SVC.best_params_)
            print('AdaClassifier - best params: ',search_ada.best_params_)
            print('LogisticRegression - best params:',search_Log.best_params_)
            print('MLPClassifier - best params: ',search_MLP.best_params_)
            print('KNeighborsClassifier - best params: ',search_KClass.best_params_)
            Gradclass_err = accuracy_score(self.y_test, search_Grad.predict(self.x_test))
            RandForclass_err = accuracy_score(self.y_test, search_rand.predict(self.x_test))
            DecTreeclass_err = accuracy_score(self.y_test, search_dec.predict(self.x_test))
            # SVCclass_err = accuracy_score(self.y_test, search_SVC.predict(self.x_test))
            adaclass_err = accuracy_score(self.y_test, search_ada.predict(self.x_test))
            LogReg_err = accuracy_score(self.y_test, search_Log.predict(self.x_test))
            MLPClass_err = accuracy_score(self.y_test, search_MLP.predict(self.x_test))
            KClass_err = accuracy_score(self.y_test, search_KClass.predict(self.x_test))
            print(f'Keras best params: {keras_grid.best_score_}, {keras_grid.best_params_}')
        else:
            Gradclass_err = accuracy_score(self.y_test, Gradclass.predict(self.x_test))
            RandForclass_err = accuracy_score(self.y_test, RandForclass.predict(self.x_test))
            DecTreeclass_err = accuracy_score(self.y_test, DecTreeclass.predict(self.x_test))
            adaclass_err = accuracy_score(self.y_test, ada_class.predict(self.x_test))
            LogReg_err = accuracy_score(self.y_test, LogReg.predict(self.x_test))
            MLPClass_err = accuracy_score(self.y_test, MLPClass.predict(self.x_test))
            KClass_err = accuracy_score(self.y_test, KClass.predict(self.x_test))
            print('GradientBoostingClassifier accuracy',Gradclass_err)
            print('RandomForestClassifier accuracy',RandForclass_err)
            print('DecisionTreeClassifier accuracy',DecTreeclass_err)
            # print('SVC accuracy',SVCclass_err)
            print('AdaClassifier accuracy',adaclass_err)
            print('LogisticRegression  accuracy',LogReg_err)
            print('MLPClassifier accuracy',MLPClass_err)
            print('KNeighborsClassifier accuracy',KClass_err)
            print("KerasClassifier: test loss, test acc:", scores)
            # print('KerasClassifier accuracy', np.mean(keras_acc))
            print('check the amount of wins and losses are in the training label data: ',self.y_train.value_counts())
    
    def prob_plots(self,col_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        prob = stats.probplot(self.x_train[col_name], dist=stats.norm, plot=ax1)
        title = f'probPlot of training data against normal distribution, feature: {col_name}'  
        ax1.set_title(title,fontsize=10)
        save_name = 'probplot_' + col_name + '.png'
        plt.savefig(join(getcwd(), 'prob_plots',save_name), dpi=200)
        plt.tight_layout()
        # plt.close()
        
    def feature_importances(self,model):
        feature_imp = pd.Series(model.feature_importances_,index=self.x_test.columns).sort_values(ascending=False)
        plt.figure(1)
        sns.barplot(x=feature_imp,y=feature_imp.index)
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title('Feature importances for - model')
        plt.show()


def main():
    start_time = time.time()
    cfb_class = cfb()
    cfb_class.input_arg()
    cfb_class.read_hyper_params()
    cfb_class.get_teams()
    cfb_class.split()
    cfb_class.machine()
    print("--- %s seconds ---" % (time.time() - start_time))
if __name__ == '__main__':
    main()
    
