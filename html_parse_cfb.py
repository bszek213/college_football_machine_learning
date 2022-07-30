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
def html_to_df_web_scrape(URL):
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
    for trb in tr_body:
        for td in trb.find_all('td'):
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
    pass_int)),
                columns =['game_result','turnovers', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'rush_att', 
                   'rush_yds', 'rush_td', 'rush_yds_per_att', 'tot_plays', 'tot_yds_per_play',
                   'first_down_pass', 'first_down_rush', 'first_down_penalty', 'first_down', 'penalty', 'penalty_yds', 'fumbles_lost',
                   'pass_int'])
    return df
#     print(df)
# html_to_df_web_scrape('https://www.sports-reference.com/cfb/schools/georgia/2021/gamelog/')