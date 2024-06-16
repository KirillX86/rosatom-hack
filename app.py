import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
from calc_scipt import calc_schedule
from tools import make_way, ledokol_way

# Загрузка данных
points_df = pd.read_excel('./data/ГрафДанные.xlsx', sheet_name='points')
edges_df = pd.read_excel('./data/ГрафДанные.xlsx', sheet_name='edges')
data = pd.read_csv('./data/рэндомное_расписание_движения_судов.csv')
ledokol = pd.read_csv('./data/рэндомное расписание ледоколов.csv', encoding='cp1251')


task = st.file_uploader("Выберите файл с task", type=["xlsx"])
start_positions = st.file_uploader("Выберите файл с start_positions", type=["xlsx"])

if not task:
    task = "data/tasks_v1_2024-06-12.xlsx"
if not start_positions:
    start_positions = "data/Ice_brakers_start_positions.xlsx"

data, ledokol = calc_schedule(
    pd.read_excel(task), #task
    pd.read_excel(start_positions) #start postioons
)

# Создание карты
m = folium.Map(location=[70, 169], zoom_start=4)


# ---------------------------------------------------------
# Добавляем стили
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

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
grouped = data.groupby('id_cудна')
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
    st_folium(m, width=700, height=600)
    st.form_submit_button('')

# --------------------------------------------------------------------------------

# Создаем диаграмму Ганта
df_gantt = data.copy()
df_gantt['start'] = df_gantt['время_прохождения']
df_gantt['end'] = df_gantt['start'].shift(-1)
# df_gantt['end'].fillna(df_gantt['start'] + pd.Timedelta(minutes=10), inplace=True)
df_gantt['ship_icebreaker'] = df_gantt.apply(lambda x: f"{x['id_cудна']} -- {x['id_ледокола']}", axis=1)

# Фильтрируем данные для диаграммы Ганта
df_gantt_filtered = df_gantt[df_gantt['sailing_status'] == 1]

# Строим диаграмму Ганта
fig = px.timeline(
    df_gantt_filtered,
    x_start="start",
    x_end="end",
    y="id_cудна",
    color="ship_icebreaker",
    labels={"start": "Начало", "end": "Конец", "id_cудна": "Корабль", "ship_icebreaker": "Корабль -- Ледокол"},
)

fig.update_yaxes(categoryorder="total ascending")
fig.update_layout(xaxis_title="Время", 
                  yaxis_title="Корабли", 
                  font=dict(family="Montserrat", size=14),
                  )

# Streamlit
st.title("Диаграмма Ганта движения кораблей по СМП")
st.plotly_chart(fig)