
import heapq
from collections import defaultdict
from typing import List, Tuple

PENALTY_FACTOR = 10.0  # used if direction penalty is enabled

def edge_key(a, b):
    return tuple(sorted((a, b)))


class CityGraph:
    def __init__(self,
                 cities,
                 edges_base,
                 weather,
                 weather_map,
                 base_mpg,
                 max_miles_per_day,
                 days,
                 clamp_tan=(-0.5, 0.5)):
        self.cities = cities
        self.edges_base = edges_base
        self.weather = weather
        self.weather_map = weather_map
        self.base_mpg = base_mpg
        self.max_miles_per_day = max_miles_per_day
        self.days = days
        self.clamp_tan = clamp_tan

        # Build adjacency list (undirected)
        self.adj = defaultdict(list)
        for e in edges_base:
            u, v, d = e["from"], e["to"], e["distance"]
            self.adj[u].append((v, d))
            self.adj[v].append((u, d))

        self.daily_edge_info = []
        self.build_daily_edge_info()
        #list of reachable cities for a given node
        self.reachable = [defaultdict(dict) for _ in range(self.days)]

    def build_daily_edge_info(self):
        # precompute per edge per day weight and gallons
        for day in range(self.days):
            info = {}
            for e in self.edges_base:
                a, b, dist,days, mpg , gallons = e["from"], e["to"], e["distance"],e['days'] , e['mpg'], e['gallons']

                # weather average for day
                wa = self.weather_map.get(self.weather[a][day].lower(), 3.0)
                wb = self.weather_map.get(self.weather[b][day].lower(), 3.0)
                avg_w = (wa + wb) / 2.0

                weight = gallons + avg_w  #initial cost

                info[edge_key(a, b)] = {
                    "distance": dist,
                    "gallons": gallons,
                    "avg_weather": avg_w,
                    "weight": weight,
                    "mpg": mpg,
                    "days":days
                }
            self.daily_edge_info.append(info)

    def get_edge_info(self, a, b, day):
        return self.daily_edge_info[day].get(edge_key(a, b))

    #Follows Dynamic memoization
    def reachable_within_day_min_cost(self, source, day):
        """
        All the cities reachable from this given using the minimum possible
        travel cost for the day
        cost is defined as fuel needed for travel + weather_risk
        :param source: current city
        :param day: today
        :return: list of possible reachable city
        """
        if self.reachable[day].get(source): #if already exists skip
            return self.reachable[day][source]

        max_allowed_distance = self.max_miles_per_day

        # PQ defines the cost
        pq: List[Tuple[float, float, str, List[str], List[Tuple[str, str]]]] = []
        heapq.heappush(pq, (0.0, 0.0, source, [source], []))
        best_cost = {}  #lowest cost entries for each city
        result = {} #reachable city

        #dijastra loop
        while pq:
            #pop pq
            cost_so_far, dist_so_far, node, path_nodes, path_edges = heapq.heappop(pq)
            # If we have cheaper already
            if node in best_cost and cost_so_far > best_cost[node] + 1e-9:
                continue
            #Mark the city as reachable
            result[node] = {"cost": cost_so_far, "distance": dist_so_far, "path": list(path_nodes),
                            "edges": list(path_edges)}
            best_cost[node] = cost_so_far

            #Explore the neighbours
            for neighbour_city, edge_distance in self.adj.get(node, []):
                #distance if decided to this city
                new_dist = dist_so_far + edge_distance
                #if not within milage skip it
                if new_dist > max_allowed_distance + 1e-9:
                    continue
                #get avg weather and fuel needed for today to travel the route
                edge_info = self.get_edge_info(node, neighbour_city, day)
                if edge_info is None: #fail-safe
                    continue
                #Cost if we travel this route
                new_cost = cost_so_far + edge_info["weight"]

                #build the route
                new_path_nodes = path_nodes + [neighbour_city]
                new_path_edges = path_edges + [(node, neighbour_city)]
                # Push to pq
                if neighbour_city not in best_cost or new_cost + 1e-9 < best_cost.get(neighbour_city, float("inf")):
                    heapq.heappush(pq, (new_cost, new_dist, neighbour_city, new_path_nodes, new_path_edges))

        #memoize the results
        self.reachable[day][source] = result

        return result
