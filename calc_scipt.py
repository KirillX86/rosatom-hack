import pandas as pd
import networkx as nx

def build_ice_graph(ice_conditon_map_df_dict):
    # ice_conditon_map_df_dict - список слварей сиитанных из файла с ледовыми условиями
    # ключ название листа
    # содержит листы lat, lon лист с датами 
    G = nx.read_graphml('data/sea_ice_cond_graph_20240612_163849.graphml')
    return G

def calc_schedule(tasks_df, ice_breakers_df):
    ship_schedule = pd.read_csv("data/рэндомное расписание движения судов.csv", encoding="cp1251")
    ice_breaker_schedule = pd.read_csv("data/рэндомное расписание ледоколов.csv", encoding="cp1251")
    return ship_schedule, ice_breaker_schedule

sheets_dict = pd.read_excel("data/IntegrVelocity.xlsx", sheet_name=None, header=None)
G = build_ice_graph(sheets_dict)

tasks_df = pd.read_excel("data/tasks_v1_2024-06-12.xlsx")
ice_breaker_df = pd.read_excel("data/Ice_brakers_start_positions.xlsx")

calc_schedule(tasks_df, ice_breaker_df)