import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from tools import make_way, ledokol_way

# Загрузка данных
points_df = pd.read_excel('./data/ГрафДанные.xlsx', sheet_name='points')
edges_df = pd.read_excel('./data/ГрафДанные.xlsx', sheet_name='edges')
data = pd.read_csv('./data/рэндомное_расписание_движения_судов.csv')
ledokol = pd.read_csv('./data/рэндомное расписание ледоколов.csv', encoding='cp1251')

# Создание карты
m = folium.Map(location=[70, 169], zoom_start=4)

# ----------------------------------------------------------
# Добавление точек на карту
for _, row in points_df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['point_name'],
        tooltip=row['point_name']
    ).add_to(m)

# Добавление ребер на карту
for _, row in edges_df.iterrows():
    start_point = points_df[points_df['point_id'] == row['start_point_id']].iloc[0]
    end_point = points_df[points_df['point_id'] == row['end_point_id']].iloc[0]
    folium.PolyLine(
        locations=[
            [start_point['latitude'], start_point['longitude']],
            [end_point['latitude'], end_point['longitude']]
        ],
        alpha=0.4,
        weight=1,
        opacity=0.9,
        color='gray'
    ).add_to(m)

# -------------------------------------------------------------------

checkboxes = {}
# sidebar
st.sidebar.title("Грузовые корабли")
grouped = data.groupby('boat_id')
for boat_name, group in grouped:
    checkboxes[boat_name] = st.sidebar.checkbox(f"Показать {boat_name}", False)

st.sidebar.title("Ледоколы")
ledokol_grouped = ledokol.groupby('id_ледокола')
for boat_name, group in ledokol_grouped:
    checkboxes[boat_name] = st.sidebar.checkbox(f"Показать {boat_name}", False)

# Отображение карты в Streamlit
st.title("Северный морской путь", )

for boat_name, group in grouped:
    if checkboxes[boat_name]:
        make_way(m, group)
        
for boat_name, group in ledokol_grouped:
    if checkboxes[boat_name]:
        ledokol_way(m, group)

with st.form(key='myfrom'):
    st_folium(m, width=700, height=450)
    st.form_submit_button('')
