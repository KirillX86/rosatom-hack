import pandas as pd
import networkx as nx
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime
import cvxpy as cp

# Тип точки, представляющей собой координаты широты и долготы, время и ID узла графа
Point = Tuple[float, float, datetime, int]

@dataclass
class CaravanPath:
    points: List[Point]

@dataclass
class ShipJoining:
    point: Point

@dataclass
class ShipPath:
    points: List[Point]

@dataclass
class TransportTask:
    destination_point: Point
    start_point: Point
    vessel_name: str
    speed: float
    type: str

@dataclass
class Icebreaker:
    name: str
    start_point: Point

@dataclass
class Caravan:
    lead_icebreaker_id: int
    caravan_path: CaravanPath
    start_point: Point
    end_point: Point
    joining_points: List[ShipJoining]
    entering_paths: List[ShipPath]
    exiting_paths: List[ShipPath]
    final_paths: List[ShipPath]
    transport_tasks: List[TransportTask]
    icebreakers: List[Icebreaker]

@dataclass
class CaravanResult:
    caravans: List[Caravan]

def calculate_weight(value, speed, type, name_ship):
    if (type == 'Нет') or (type == 'Ice1') or (type == 'Ice2') or (type == 'Ice3'):
        if value < 10:
            return -1
        if 10 <= value <= 14.5:
            return -1
        if 14.5 < value <= 19.5:
            return 0
        if value > 19.5:
            return speed

    if (type == 'Arc 4') or (type == 'Arc 5') or (type == 'Arc 6'):
        if value < 10:
            return -1
        if 10 <= value <= 14.5:
            return 0.7 * speed
        if 14.5 < value <= 19.5:
            return 0.8 * speed
        if value > 19.5:
            return speed

    if (type == 'Arc 7'):
        if value < 10:
            return -1
        if 10 <= value <= 14.5:
            return 0.15 * speed
        if 14.5 < value <= 19.5:
            return 0.6 * speed
        if value > 19.5:
            return speed

    if (type == 'Arc 9') and ((name_ship == 'Ямал') or (name_ship == '50 лет Победы')):
        if value < 10:
            return -1
        if 10 <= value <= 14.5:
            return value
        if 14.5 < value <= 19.5:
            return value
        if value > 19.5:
            return speed

    if (type == 'Arc 9') and ((name_ship == 'Вайгач') or (name_ship == 'Таймыр')):
        if value < 10:
            return -1
        if 10 <= value <= 14.5:
            return 0.75 * value
        if 14.5 < value <= 19.5:
            return 0.9 * value
        if value > 19.5:
            return speed

def dijkstra(graph, start, speed, type, name, date):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    predecessors = {}

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex].items():
            if graph.nodes[neighbor][date] >= 10:
                dist = abs(weight['weight'] / (calculate_weight(graph.nodes[neighbor][date], int(speed), type, name) * 1.852))
                distance = current_distance + dist

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, neighbor))

    return distances, predecessors

def create_transport_tasks(df: pd.DataFrame) -> List[TransportTask]:
    transport_tasks = []
    for _, row in df.iterrows():
        destination_point = (row['destination_lat'], row['destination_lon'], pd.to_datetime(row['destination_time']), row['destination_node_id'])
        start_point = (row['start_lat'], row['start_lon'], pd.to_datetime(row['start_time']), row['start_node_id'])
        vessel_name = row['vessel_name']
        speed = row['speed']
        type = row['type']
        
        transport_task = TransportTask(
            destination_point=destination_point,
            start_point=start_point,
            vessel_name=vessel_name,
            speed=speed,
            type=type
        )
        transport_tasks.append(transport_task)
    
    return transport_tasks

def create_icebreakers(df: pd.DataFrame) -> List[Icebreaker]:
    icebreakers = []
    for _, row in df.iterrows():
        start_point = (row['start_lat'], row['start_lon'], pd.to_datetime(row['start_time']), row['start_node_id'])
        icebreaker = Icebreaker(
            name=row['name'],
            start_point=start_point
        )
        icebreakers.append(icebreaker)
    
    return icebreakers

def heuristic_optimal_point(graph, tasks, icebreakers, eps, delta, N, date):
    results = []
    
    for task in tasks:
        task_results = []

        # Эвристика 1: Оптимальная точка выхода из каравана последнего судна
        ship_distances, ship_predecessors = dijkstra(graph, task.start_point[3], task.speed, task.type, task.vessel_name, date)
        distance_to_destination, path_to_destination = ship_distances[task.destination_point[3]], []

        current_vertex = task.destination_point[3]
        while current_vertex != task.start_point[3]:
            path_to_destination.append(current_vertex)
            current_vertex = ship_predecessors.get(current_vertex)
        path_to_destination.append(task.start_point[3])
        path_to_destination.reverse()

        task_results.append((distance_to_destination, path_to_destination))

        # Эвристика 2: Оптимальная точка выхода из каравана по пути следования каравана
        for i in range(N + 1):
            threshold = eps + i * delta
            for vertex, dist in ship_distances.items():
                if abs(ship_distances[task.destination_point[3]] - dist) <= threshold:
                    task_results.append((dist, path_to_destination))

        # Эвристика 3: Оптимальная точка присоединения судна к каравану
        for icebreaker in icebreakers:
            ice_distances, ice_predecessors = dijkstra(graph, icebreaker.start_point[3], icebreaker.speed, icebreaker.type, icebreaker.name, date)
            distance_to_start, path_to_start = ice_distances[task.start_point[3]], []

            current_vertex = task.start_point[3]
            while current_vertex != icebreaker.start_point[3]:
                path_to_start.append(current_vertex)
                current_vertex = ice_predecessors.get(current_vertex)
            path_to_start.append(icebreaker.start_point[3])
            path_to_start.reverse()

            task_results.append((distance_to_start, path_to_start))

        # Эвристика 4: Оптимальная точка формирования каравана
        for i in range(N + 1):
            threshold = eps + i * delta
            for vertex, dist in ship_distances.items():
                if abs(ship_distances[task.destination_point[3]] - dist) <= threshold:
                    task_results.append((dist, path_to_destination))

        results.append(task_results)
    
    return results

def generate_lp_model(results, icebreakers, max_caravans=3):
    num_results = len(results)
    num_icebreakers = len(icebreakers)

    x = cp.Variable(num_results, integer=True)  # Переменные для выбора маршрутов

    constraints = []

    # Ограничение: каждый ледокол может вести не более одного каравана в один момент времени
    for i in range(num_icebreakers):
        constraints.append(cp.sum(x) <= 1)

    # Ограничение: каждое судно может проводиться только одним караваном в один момент времени
    for j in range(num_results):
        constraints.append(cp.sum(x) <= 1)

    # Ограничение: не более max_caravans караванов
    constraints.append(cp.sum(x) <= max_caravans)

    # Целевая функция: минимизировать время ведения караванов и время подплытия судов к ним из пунктов отправления и отправления их к пунктам назначения
    objective = cp.Minimize(cp.sum(x))

    problem = cp.Problem(objective, constraints)
    return problem, x

def calc_schedule(tasks_df: pd.DataFrame, ice_breakers_df: pd.DataFrame, eps: float = 1, delta: float = 6, N: int = 3, graphml_path: str = "data/sea_ice_cond_graph_20240612_163849.graphml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Чтение графа из файла GraphML
    G = nx.read_graphml(graphml_path)

    transport_tasks = create_transport_tasks(tasks_df)
    icebreakers = create_icebreakers(ice_breakers_df)

    # Разделение задач на группы по 3 судна
    task_groups = [transport_tasks[i:i + 3] for i in range(0, len(transport_tasks), 3)]
    all_caravans = []
    icebreaker_schedule = []

    for group in task_groups:
        # Поиск оптимальных точек для всех задач в группе
        results = heuristic_optimal_point(G, group, icebreakers, eps, delta, N, 'date')

        # Генерация модели линейного программирования
        problem, x = generate_lp_model(results, icebreakers)

        # Решение задачи линейного программирования
        problem.solve(solver=cp.GLPK_MI)

        # Формирование караванов на основе результатов линейного программирования
        for i in range(len(group)):
            if x.value[i] > 0.5:
                optimal_result = results[i]
                distance, path = optimal_result[0]
                final_paths = [ShipPath(points=[(G.nodes[node]['lat'], G.nodes[node]['lon'], datetime.now(), node) for node in path])]
                entering_paths = final_paths
                exiting_paths = final_paths
                joining_points = [ShipJoining(point=(G.nodes[path[0]]['lat'], G.nodes[path[0]]['lon'], datetime.now(), path[0]))]

                caravan = Caravan(
                    lead_icebreaker_id=icebreakers[0].name,  # ID ведущего ледокола
                    caravan_path=CaravanPath(points=[]),  # Путь каравана будет определен позже
                    start_point=final_paths[0].points[0] if final_paths else (0, 0, datetime.now(), 0),
                    end_point=final_paths[-1].points[-1] if final_paths else (0, 0, datetime.now(), 0),
                    joining_points=joining_points,
                    entering_paths=entering_paths,
                    exiting_paths=exiting_paths,
                    final_paths=final_paths,
                    transport_tasks=group,
                    icebreakers=icebreakers
                )

                all_caravans.append(caravan)

                # Формирование расписания движения ледоколов
                for path in final_paths:
                    for i, point in enumerate(path.points):
                        latitude, longitude, timestamp, _ = point
                        if i < len(path.points) - 1:
                            icebreaker_schedule.append([caravan.lead_icebreaker_id, latitude, longitude, timestamp.isoformat(), group[0].vessel_name, group[1].vessel_name if len(group) > 1 else '', group[2].vessel_name if len(group) > 2 else ''])
                        else:
                            icebreaker_schedule.append([caravan.lead_icebreaker_id, latitude, longitude, timestamp.isoformat(), '', '', ''])
    
    caravan_df = convert_caravan_result_to_df(CaravanResult(caravans=all_caravans))
    icebreaker_schedule_df = pd.DataFrame(icebreaker_schedule, columns=['icebreaker_id', 'latitude', 'longitude', 'timestamp', 'vessel_id_1', 'vessel_id_2', 'vessel_id_3'])
    
    return caravan_df, icebreaker_schedule_df

def convert_caravan_result_to_df(caravan_result: CaravanResult) -> pd.DataFrame:
    data = []
    for caravan in caravan_result.caravans:
        caravan_id = caravan.lead_icebreaker_id
        for path in caravan.final_paths:
            for i, point in enumerate(path.points):
                vessel_id = caravan.transport_tasks[0].vessel_name
                latitude, longitude, timestamp, _ = point
                sailing_status = 1 if i < len(path.points) - 1 else 0
                icebreaker_id = caravan.lead_icebreaker_id if sailing_status == 1 else ''
                data.append([vessel_id, latitude, longitude, timestamp.isoformat(), sailing_status, icebreaker_id, caravan_id])
    
    df = pd.DataFrame(data, columns=['vessel_id', 'latitude', 'longitude', 'timestamp', 'sailing_status', 'icebreaker_id', 'caravan_id'])
    return df

def build_ice_graph(ice_condition_map_df_dict: Dict[str, pd.DataFrame]) -> nx.Graph:
    G = nx.read_graphml('data/sea_ice_cond_graph_20240612_163849.graphml')
    for sheet_name, df in ice_condition_map_df_dict.items():
        if sheet_name == 'lat' or sheet_name == 'lon':
            continue
        for _, row in df.iterrows():
            node_id = row['node_id']
            date = row['date']
            ice_condition = row['ice_condition']
            G.nodes[node_id][date] = ice_condition
    return G
