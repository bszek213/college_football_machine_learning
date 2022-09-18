"""
College football game regressor
@author: brianszekely
"""
import pandas as pd
from os import exists, join
from os import getcwd
import yaml
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
        pass