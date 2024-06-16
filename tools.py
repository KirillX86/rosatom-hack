import pandas as pd
import folium
import random

colors = ['red', 'pink', 'lightred', 'purple', 'darkblue', 'lightgray', 'blue', 'beige', 'lightgreen', 'white', 'green', 'darkred', 'cadetblue', 'gray', 'darkgreen', 'orange', 'darkpurple', 'lightblue']

def make_way(mapp, group):
    color = random.choice(colors)
    # Добавляем точки на карту
    for idx, row in group.iterrows():
        popup_text = f"Время: {'<br>'.join(row['время_прохождения'][:-3].split('T'))}"
        if pd.notna(row['id_ледокола']):
            popup_text += f"<br>Ледокол: {row['id_ледокола']}"
        folium.Marker(
            location=[row["широта"], row["долгота"] + 0.1],
            popup=popup_text,
            icon=folium.Icon(color=color, prefix='fa', icon='sailboat' )
        ).add_to(mapp)
        
    # Добавляем линии между точками
    for i in range(len(group) - 1):
        points = group[['широта', 'долгота']].iloc[i:i+2].values.tolist()
        folium.PolyLine(points, color=color, weight=2.5, opacity=1).add_to(mapp)

def ledokol_way(mapp, group):
    color = 'black'
    # Добавляем точки на карту
    for idx, row in group.iterrows():
        popup_text = row['id_ледокола']
        popup_text += f"<br>Время: {'<br>'.join(row['время_прохождения'][:-3].split('T'))}"

        folium.Marker(
            location=[row["широта"], row["долгота"]],
            popup=popup_text,
            icon=folium.Icon(color=color, prefix='fa', icon='ship' )
        ).add_to(mapp)
        
        # Добавляем линии между точками
    for i in range(len(group) - 1):
        points = group[['широта', 'долгота']].iloc[i:i+2].values.tolist()
        folium.PolyLine(points, color=color, weight=15, opacity=0.35).add_to(mapp)