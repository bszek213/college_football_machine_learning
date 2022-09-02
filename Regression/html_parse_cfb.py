# -*- coding: utf-8 -*-
"""
html parse code - cfb
"""# -*- coding: utf-8 -*-
"""
html parse code - cfb
"""
import requests
from bs4 import BeautifulSoup
from pandas import DataFrame
import cfbd
from numpy import nan
# from cfbd.rest import ApiException
def html_to_df_web_scrape(URL,team,year):
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = 'UK4ikHBmxuHDyMlNngTZS8sokyl8Kr4FExP2NRb9G8qaFOUrUhX3xy6+OxQv4oEX'
    configuration.api_key_prefix['Authorization'] = 'Bearer'
    api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    api_game = cfbd.GamesApi(cfbd.ApiClient(configuration))
    # URL EXAMPLE: URL = "https://www.sports-reference.com/cfb/schools/georgia/2021/gamelog/"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    table = soup.find(id="div_offense")
    tbody = table.find('tbody')
    tr_body = tbody.find_all('tr')
    game_result = []
    turnovers = []
    pass_cmp = []
    pass_att = []
    pass_yds = []
    pass_td = []
    rush_att = []
    rush_yds = []
    rush_td = []
    rush_yds_per_att = []
    tot_plays = []
    tot_yds_per_play = []
    first_down_pass = []
    first_down_rush = []
    first_down_penalty = []
    first_down = []
    penalty = []
    penalty_yds = []
    fumbles_lost = []
    pass_int = []
    havoc = []
    for trb in tr_body:
        for td in trb.find_all('td'):
            if td.get('data-stat') == "opp_name":
                if '*' in td.get_text():
                    text_data = td.get_text().replace('*','')
                else:
                    text_data = td.get_text()
                print('opp:',text_data)
                print('team:',team)
                if text_data == 'Miami (FL)':
                    text_data = 'Miami'
                elif text_data == 'Mississippi':
                    text_data = 'Ole Miss'
                elif text_data == 'Louisiana State':
                    text_data = 'LSU'
                elif text_data == 'Nevada-Las Vegas':
                    text_data = 'UNLV'
                elif text_data == 'Bowling Green State':  
                    text_data = 'Bowling Green'
                elif text_data == 'UTSA':
                    text_data = 'UT San Antonio'
                elif text_data   == 'Brigham Young':
                    text_data = 'BYU'
                elif text_data   == 'Southern California':
                    text_data = 'USC'
                elif text_data   == 'Massachusetts': 
                    text_data = 'Umass'
                elif text_data   == 'Central Florida': 
                    text_data = 'UCF'
                elif text_data   == 'North Carolina State': 
                    text_data = 'NC State'
                elif text_data == 'Alabama-Birmingham':
                    text_data = 'UAB'
                elif text_data == 'Southern Methodist':
                    text_data = 'SMU'
                elif text_data == 'Middle Tennessee State':
                    text_data = 'Middle Tennessee'
                elif text_data == 'San Jose State':
                    text_data = 'San José State'
                else:
                    text_data = text_data
                if '-' in text_data:
                    text_data = text_data.replace('-',' ')
                if '-' in team:
                    team = team.replace('-',' ')
                #Execptions since no one can chose a syntax and stick to it
                if text_data == 'Arkansas Pine Bluff':
                    text_data = 'Arkansas-Pine Bluff'
                if text_data == 'Bethune Cookman':
                    text_data = 'Bethune-Cookman'
                if text_data == 'Gardner Webb':
                    text_data = 'Gardner-Webb'
                #Fix team names
                if team == 'alabama birmingham':
                    team = 'UAB'
                if team == 'bowling green state':
                    team = 'Bowling Green'
                if team == 'brigham young':
                    team = 'BYU'
                if team == 'central florida':
                    team = 'UCF'
                if '*' in text_data:
                    text_data = text_data.replace('*','')
                print('opp:',text_data)
                print('team:',team)
                api_response = api_instance.get_games(year, season_type='regular',
                                                      
                                                      home=text_data, away=team,  #add a if statement here to say if null switch home and away
                                                      )
                if not api_response:
                    api_response = api_instance.get_games(year, season_type='regular',
                                                          
                                                          home=team, away=text_data,  #add a if statement here to say if null switch home and away
                                                          )
                if not api_response:
                    api_response = api_instance.get_games(year, season_type='postseason',
                                                          
                                                          home=team, away=text_data,  #add a if statement here to say if null switch home and away
                                                          )
                if not api_response:
                    api_response = api_instance.get_games(year, season_type='postseason',
                                                          
                                                          home=text_data, away=team,  #add a if statement here to say if null switch home and away
                                                          )
                print(api_response)
                api_response_2 = api_game.get_advanced_box_score(api_response[0].id)
                if api_response_2.teams['havoc']:
                    if api_response_2.teams['havoc'][0]['team'].capitalize() == team.capitalize():
                        havoc.append(api_response_2.teams['havoc'][0]['total'])
                    else:
                        havoc.append(api_response_2.teams['havoc'][1]['total'])
                    print('=========================================')
                    print(api_response_2.teams['havoc'][0]['total'])
                    print(api_response_2.teams['havoc'][1]['total'])
                    print('=========================================')
                else:
                    havoc.append(nan)
            if td.get('data-stat') == "game_result":
                game_result.append(td.get_text())
            if td.get('data-stat') == "turnovers":
                turnovers.append(td.get_text())
            if td.get('data-stat') == "pass_cmp":
                pass_cmp.append(td.get_text())
            if td.get('data-stat') == "pass_att":
                pass_att.append(td.get_text())
            if td.get('data-stat') == "pass_cmp_pct":
                pass_att.append(td.get_text())
            if td.get('data-stat') == "pass_yds":
                pass_yds.append(td.get_text())
            if td.get('data-stat') == "pass_yds":
                pass_yds.append(td.get_text())
            if td.get('data-stat') == "pass_td":
                pass_td.append(td.get_text())
            if td.get('data-stat') == "rush_att":
                rush_att.append(td.get_text())
            if td.get('data-stat') == "rush_yds":
                rush_yds.append(td.get_text())
            if td.get('data-stat') == "rush_yds_per_att":
                rush_yds_per_att.append(td.get_text())
            if td.get('data-stat') == "rush_td":
                rush_td.append(td.get_text())
            if td.get('data-stat') == "tot_plays":
                tot_plays.append(td.get_text())
            if td.get('data-stat') == "tot_yds_per_play":
                tot_yds_per_play.append(td.get_text())
            if td.get('data-stat') == "first_down_pass":
                first_down_pass.append(td.get_text())
            if td.get('data-stat') == "first_down_rush":
                first_down_rush.append(td.get_text())
            if td.get('data-stat') == "first_down_penalty":
                first_down_penalty.append(td.get_text())
            if td.get('data-stat') == "first_down":
                first_down.append(td.get_text())
            if td.get('data-stat') == "penalty":
                penalty.append(td.get_text())
            if td.get('data-stat') == "penalty_yds":
                penalty_yds.append(td.get_text())
            if td.get('data-stat') == "pass_int":
                pass_int.append(td.get_text())
            if td.get('data-stat') == "fumbles_lost":
                fumbles_lost.append(td.get_text())


    df = DataFrame(list(zip(game_result,turnovers,pass_cmp,pass_att,pass_yds,
    pass_td,
    rush_att,
    rush_yds,
    rush_td, 
    rush_yds_per_att,
    tot_plays,
    tot_yds_per_play,
    first_down_pass,
    first_down_rush,
    first_down_penalty,
    first_down,
    penalty,
    penalty_yds,
    fumbles_lost,
    pass_int,havoc)),
                columns =['game_result','turnovers', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'rush_att', 
                   'rush_yds', 'rush_td', 'rush_yds_per_att', 'tot_plays', 'tot_yds_per_play',
                   'first_down_pass', 'first_down_rush', 'first_down_penalty', 'first_down', 'penalty', 'penalty_yds', 'fumbles_lost',
                   'pass_int','havoc'])
    return df

def cbfd(school):
    configuration = cfbd.Configuration()
    configuration.api_key['Authorization'] = 'UK4ikHBmxuHDyMlNngTZS8sokyl8Kr4FExP2NRb9G8qaFOUrUhX3xy6+OxQv4oEX'
    configuration.api_key_prefix['Authorization'] = 'Bearer'
    api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    api_response = api_instance.get_games(2022, season_type='regular', 
                                          home='Northwestern', away='Nebraska',  #add a if statement here to say if null switch home and away
                                          )
    api_game = cfbd.GamesApi(cfbd.ApiClient(configuration))
    api_response_2 = api_game.get_advanced_box_score(api_response[0].id)
    # api_instance = cfbd.GamesApi(cfbd.ApiClient(configuration))
    print('=========================================')
    print(api_response_2.teams['havoc'][0]['total'])
    print(api_response_2.teams['havoc'][1]['total'])
    print('=========================================')
#     print(df)
# html_to_df_web_scrape('https://www.sports-reference.com/cfb/schools/georgia/2021/gamelog/')