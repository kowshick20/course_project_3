import heapq
import math
import sys
from collections import defaultdict
from typing import List, Tuple, Any, Dict, Optional

from numpy.f2py.crackfortran import sourcecodeform
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from CityGraph import CityGraph


# def build_initial_route(cities, costs):
#     scores = []
#     for city in cities:
#         s = sum(costs.get((city, c), float('inf')) for c in cities if c != city)
#         # s = sum(costs[(city,c)] for c in cities if c!=city)
#         scores.append((s, city))
#     scores.sort()
#     start_city = scores[0][1]
#     route = [start_city]
#     unvisited = set(cities) - {start_city}  # unvisited city list
#
#     while unvisited:
#         prev_city = route[-1]
#         next_city = min(unvisited, key=lambda x: costs.get((prev_city, x), float("inf")))
#         route.append(next_city)  # take a move
#         unvisited.remove(next_city)  # mark as visited
#
#     return route


def build_initial_route(cities, costs):

    def safe_cost(a, b):
        entry = costs.get((a, b), float("inf"))
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, dict) and "cost" in entry:
            return float(entry["cost"])
        return float("inf")

    # -------------------------
    # 1. Find best start city
    # -------------------------
    best_city = None
    best_sum = float("inf")

    for city in cities:
        total = 0.0
        for other in cities:
            if other != city:
                total += safe_cost(city, other)

        if total < best_sum:
            best_sum = total
            best_city = city

    # If no valid start city (all INF), pick the first city
    if best_city is None:
        # You can also raise an error here if this shouldn't happen:
        # raise ValueError("All pairwise costs are infinite!")
        best_city = next(iter(cities))

    start_city = best_city

    # -------------------------
    # 2. Pure greedy route
    # -------------------------
    route = [start_city]
    unvisited = set(cities)
    unvisited.remove(start_city)  # **safe because start_city is never None now**

    while unvisited:
        current = route[-1]

        next_city = min(
            unvisited,
            key=lambda c: safe_cost(current, c)
        )

        route.append(next_city)
        unvisited.remove(next_city)
        print("building",route)

    return route



def improve_path(route, costs, shots=200):
    """
    Optimize the travel route using pairwise costs in 'costs' dict.
    Uses 2-opt-like swaps to reduce total cost.
    """
    def safe_cost(a, b):
        entry = costs.get((a, b), float("inf"))
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, dict) and "cost" in entry:
            return float(entry["cost"])
        return float("inf")

    def full_path_cost(path):
        total = 0.0
        for i in range(len(path) - 1):
            total += safe_cost(path[i], path[i + 1])
        return total

    best_path = route[:]
    best_cost = full_path_cost(best_path)
    found_better = True
    attempt = 0
    n = len(route)

    while found_better and attempt < shots:
        found_better = False
        attempt += 1

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # original edges
                a, b = best_path[i - 1], best_path[i]
                c, d = best_path[j], best_path[j + 1]

                old_cost = safe_cost(a, b) + safe_cost(c, d)
                new_cost = safe_cost(a, c) + safe_cost(b, d)

                if new_cost < old_cost:
                    # perform 2-opt swap
                    best_path[i:j + 1] = best_path[i:j + 1][::-1]
                    best_cost += new_cost - old_cost
                    found_better = True
                    break  # restart search after improvement

            if found_better:
                break

    return best_path


# def improve_path(route, cost, shots=200):
#     """
#     Optimize the travel route by making  cost-effective, by interchanging the
#     source and destination cities, if it reduces the cost
#     :param route: initial path to travel all cities
#     :param cost: initial cost
#     :param shots: maximum iterations allowed
#     :return: new path
#     """
#
#     def cost_between(current, dest):
#         """Return numeric cost between cities."""
#         if (current, dest) in cost:
#             val = cost[(current, dest)]
#         elif (dest, current) in cost:
#             val = cost[(dest, current)]
#         else:
#             return float("inf")  # no path
#
#         # Always extract numeric cost
#         return val
#
#     def full_path_cost(path):
#         total_cost = 0.0
#         for i in range(len(path) - 1):
#             total_cost += cost_between(path[i], path[i + 1])
#         return total_cost
#
#     best_path = route[:]
#     best_cost = full_path_cost(best_path)  # total cost
#     found_better_path = True
#     attempt = 0
#     n = len(route)
#
#     while found_better_path and attempt < shots:
#         found_better_path = False
#         attempt += 1
#
#         for start in range(1, n - 2):
#             for des in range(start + 1, n - 1):
#                 # remove old path and connect new two cities
#                 before_starting = best_path[start - 1]
#                 start_city = best_path[start]
#                 end_city = best_path[des]
#                 after_end = best_path[des + 1]
#
#                 # old path to be removed
#                 old_cost = (
#                         cost_between(before_starting, start_city) +
#                         cost_between(end_city, after_end)
#                 )
#                 # new path to be added
#                 new_cost = (
#                         cost_between(before_starting, end_city) +
#                         cost_between(start_city, after_end)
#                 )
#
#                 cost_difference = new_cost - old_cost
#
#                 if cost_difference < 0:  # if cheaper
#                     # flip path
#                     new_path = (
#                             best_path[:start] +
#                             best_path[start:des + 1][::1] +
#                             best_path[des + 1:]
#                     )
#                     # update path and cost
#                     best_path = new_path
#                     best_cost += cost_difference
#                     found_better_path = True
#                     break  # search again
#
#             if found_better_path:
#                 break
#
#         return best_path


class Astar:
    def __init__(self, graph: CityGraph, coordinates, max_days: int = 30,
                 penalty_factor: float = 10, use_direction_penalty: bool = True):
        self.usa_map = graph  # complete USA Cities following EST
        self.max_days = graph.days  # Max available days of data
        self.penalty_factor = penalty_factor  # penalty of moving in wrong direction
        self.use_direction_penalty = use_direction_penalty
        self.pair_cache = {}
        self.coordinates = coordinates  # latitude and longitude coordinate of each city
        # Meta data JSON
        self.meta_data = {
            "daily": defaultdict(lambda: {"path": [], "distance": 0.0, "fuel": 0.0, "risk": 0.0, "edges": []}),
            "total": {"distance": 0.0, "fuel": 0.0, "risk": 0.0, "days_travelled": 0, "total_path": []}}

    ####TODO: Weak huirastic need a stronger one
    # def heuristic(self, current_place: str, destination: str) -> float:
    #     best_path = float("inf")  # initialize with infinity
    #
    #     # check if source and dest are directly connected
    #     for city in self.usa_map.edges_base:
    #         if (city["from"] == current_place and city["to"] == destination) or (
    #                 city["from"] == destination and city["to"] == current_place):
    #             best_path = min(best_path, city["distance"])
    #
    #     if best_path == float("inf"):  # save from divide by inf error
    #         return 0.0
    #
    #     return best_path / self.usa_map.base_mpg
    dest_reach_day=0
    def search(self, current: str, destination: str,start_day:int=0) -> Optional[Dict[str, Any]]:
        """
        A* start algorithm
        :param current: current city
        :param destination: destination city
        :return: metadata JSON if when reached else empty
        """
        # reset
        self.meta_data = {
            "daily": defaultdict(lambda: {"path": [], "distance": 0.0, "fuel": 0.0, "risk": 0.0, "edges": []}),
            "total": {"distance": 0.0, "fuel": 0.0, "risk": 0.0, "days_travelled": 0, "total_path": []}}

        # priority q
        # h,g,current_city,days_taken, path
        open_pq: List[Tuple[float, float, str, int, List[str]]] = []
        starting_hur = 0.0  # initial h function


        heapq.heappush(open_pq, (starting_hur, 0.0, current, start_day, [current]))  # push start city into pq

        best_route = {}  # Best route to take on the given day

        # Until empty
        while open_pq:
            # pop first element
            hur, g, current, day, path = heapq.heappop(open_pq)
            if current == destination:  # reached destination
                self.meta_data["total"]["total_path"] = path  # path taken to reach destination
                self.meta_data["total"]["days_travelled"] = day  # days taken to reach the destination
                self.populate_meta_data(path,start_day)  # generate the meta_data

                return {
                    "path": path,
                    "cost": g,
                    "days_travelled": day,
                    "meta_data": self.meta_data
                }

            if day >= self.max_days:  # not able to complete travel within given days
                continue

            key = (current, day)
            # if we have a better day to travel this route? skip today
            if key in best_route and g > best_route[key] + 1e-9:
                continue
            best_route[key] = g  # best route for the day to travel

            # Cities reachable within today


            reachable = self.usa_map.reachable_within_day_min_cost(current, day)
            #print(reachable)
            for dest, info in reachable.items():
                if dest == current:
                    continue

                trans_cost = info["cost"]  # cost taken to reach
                first_hop = info["edges"][0] if info["edges"] else (current, dest)  # Paths to be taken
                next_city = first_hop[1]  # immediate next city
                # penalize if the movement is opposite to the goal direction
                # direction_penalty = self.direction_penalty(current, next_city, destination)
                h = self.heuristic(current, next_city, destination)  # get the heuristic data
                # total cost is the cost to travel and the direction penalty
                # total_cost = trans_cost + direction_penalty
                # new accumulated cost
                new_g = g + trans_cost  # g function is added with cost to reach
                new_day = day + 1  # next day
                # h = self.heuristic(dest, destination)
                new_f = new_g + h  # f=g+h
                # include the multi hop path in final path
                new_path = path + info["path"][1:]  # skip in initial node, which is the current node
                # check for better day
                key2 = (dest, new_day)
                if key2 in best_route and new_g >= best_route[key2] - 1e-9:
                    continue
                heapq.heappush(open_pq, (new_f, new_g, dest, new_day, new_path))  # push the element

        return None  # if goal cannot be reached







    # def populate_meta_data(self, final_path,day, max_wait_days=2):
    #     """
    #     Given final path this tries to reconstruct/backtracks the travel per day
    #     based on the travel constraints
    #     :param final_path: final path to travel all the cities
    #     """
    #     remaining_cities = list(final_path)
    #     current_day = 0
    #     current_city = remaining_cities[0]
    #     i = 1  # index of next city to travel
    #     # path covered or ran out of days
    #     while i < len(remaining_cities) and current_day < 30:
    #         #print(current_day)
    #         best_travel_choice = None  #store best travel day/route
    #
    #         #lookahead the numer of days to see if the cost improves,
    #         #possible of a pit stop
    #         for wait in range(max_wait_days + 1):
    #             #look ahead of path costs
    #             day_to_check = current_day + wait
    #             #cannot get passed the max days
    #             if day_to_check >= self.usa_map.days:
    #                 continue
    #
    #             #get cities that can be reached today
    #             reachable = self.usa_map.reachable_within_day_min_cost(
    #                 current_city, day_to_check)
    #
    #             #Move as many cities as possible today (greedy approach)
    #             for j in range(i, len(remaining_cities)):
    #                 possible_dest = remaining_cities[j] #possible destination
    #                 #if reachable, then travel
    #                 if possible_dest in reachable:
    #                     metadata = reachable[possible_dest]
    #                     prefix_nodes = metadata["path"]
    #
    #                     #check if the route matches the towards the final path by A*
    #                     if prefix_nodes == remaining_cities[i - 1:j + 1]:
    #                         cost_today = metadata["cost"]
    #
    #                         #Stay the day in the city
    #                         if (best_travel_choice is None) or (cost_today < best_travel_choice["cost"]):
    #                             #add a day record that you stayed
    #
    #                             day_record = self.meta_data["daily"][current_day]
    #                             day_record["path"] = [current_city]
    #                             day_record["distance"] = 0.0
    #                             day_record["fuel"] = 0.0
    #                             day_record["risk"] = 0.0
    #                             day_record["edges"] = []
    #                             best_travel_choice = {
    #                                 "dest": possible_dest,
    #                                 "prefix_len": j - i + 1,
    #                                 "metadata": metadata,
    #                                 "wait_days": wait,
    #                                 "cost": cost_today,
    #                                 "day": day_to_check
    #                             }
    #
    #         if best_travel_choice is None:
    #             #if max days will reach, if you wait? forced to take the next best route
    #             next_city = remaining_cities[i]
    #             reachable = (self.usa_map.reachable_within_day_min_cost
    #                          (current_city, current_day))
    #             if next_city in reachable:
    #                 metadata = reachable[next_city]
    #                 best_travel_choice = {
    #                     "dest": next_city,
    #                     "prefix_len": 1,
    #                     "metadata": metadata,
    #                     "wait_days": 0,
    #                     "cost": metadata["cost"],
    #                     "day": current_day
    #                 }
    #             else:
    #                 #cannot travel
    #                 break
    #
    #         #Record this day's travel metadata
    #         day_record = self.meta_data["daily"][best_travel_choice["day"]]
    #         day_record["path"] = best_travel_choice["metadata"]["path"]
    #         day_record["distance"] = best_travel_choice["metadata"]["distance"]
    #         day_record["fuel"] = 0.0
    #         day_record["risk"] = 0.0
    #         day_record["edges"] = []
    #
    #         #get fuel,weather and mph data from the graph
    #         for (u, v) in best_travel_choice["metadata"]["edges"]:
    #             edge_info = self.usa_map.get_edge_info(u, v, best_travel_choice["day"])
    #             if edge_info is None:
    #                 continue
    #             day_record["fuel"] += edge_info["gallons"]
    #             #day_record["risk"] += edge_info["avg_weather"]
    #             day_record["edges"].append({
    #                 "from": u, "to": v, "distance": edge_info["distance"],
    #                 "gallons": edge_info["gallons"],
    #                 #"avg_weather": edge_info["avg_weather"],
    #                 "weight": edge_info["weight"]
    #             })
    #
    #             self.meta_data["total"]["distance"] += edge_info["distance"]
    #             self.meta_data["total"]["fuel"] += edge_info["gallons"]
    #             #self.meta_data["total"]["risk"] += edge_info["avg_weather"]
    #
    #         #increment to the next
    #         current_city = best_travel_choice["dest"]
    #         i += best_travel_choice["prefix_len"]
    #         current_day =  best_travel_choice["day"] + 1  #day after travel
    #
    #     self.meta_data["total"]["days_travelled"] = current_day

#TODO:AI
    # def populate_meta_data(self, final_path, max_wait_days=1):
    #     """
    #     Reconstruct the day-by-day travel plan for final_path.
    #     This version CORRECTLY handles waits and ensures current_day
    #     always advances by wait_days + 1 for each chosen move.
    #     """
    #     remaining_cities = list(final_path)
    #     current_day = 0
    #     current_city = remaining_cities[0]
    #     i = 1  # index of the next city to visit
    #
    #     # stop if we've covered all cities or run out of days
    #     while i < len(remaining_cities) and current_day < self.usa_map.days:
    #         best_travel_choice = None  # holds the chosen (wait_days, dest, metadata, prefix_len, cost)
    #
    #         # LOOKAHEAD: try 0..max_wait_days ahead and pick the cheapest valid option
    #         for wait in range(max_wait_days + 1):
    #             day_to_check = current_day + wait
    #             if day_to_check >= self.usa_map.days:
    #                 continue  # can't check past available days
    #
    #             # compute reachable cities for that candidate travel day
    #             reachable = self.usa_map.reachable_within_day_min_cost(current_city, day_to_check)
    #
    #             # try to match as many consecutive cities from remaining_cities as possible
    #             for j in range(i, len(remaining_cities)):
    #                 possible_dest = remaining_cities[j]
    #                 if possible_dest not in reachable:
    #                     continue
    #
    #                 metadata = reachable[possible_dest]
    #                 prefix_nodes = metadata["path"]
    #
    #                 # Ensure the in-day path matches the required segment of final_path
    #                 if prefix_nodes == remaining_cities[i - 1: j + 1]:
    #                     cost_today = metadata["cost"]
    #                     # choose the option with the lowest cost (tie-breaking arbitrary)
    #                     if (best_travel_choice is None) or (cost_today < best_travel_choice["cost"]):
    #                         best_travel_choice = {
    #                             "dest": possible_dest,
    #                             "prefix_len": j - i + 1,
    #                             "metadata": metadata,
    #                             "wait_days": wait,
    #                             "cost": cost_today
    #                         }
    #
    #         # If lookahead found nothing, try the immediate next city today (fallback)
    #         if best_travel_choice is None:
    #             next_city = remaining_cities[i]
    #             reachable_today = self.usa_map.reachable_within_day_min_cost(current_city, current_day)
    #             if next_city in reachable_today:
    #                 best_travel_choice = {
    #                     "dest": next_city,
    #                     "prefix_len": 1,
    #                     "metadata": reachable_today[next_city],
    #                     "wait_days": 0,
    #                     "cost": reachable_today[next_city]["cost"]
    #                 }
    #             else:
    #                 # Nothing reachable today and lookahead didn't help — stop
    #                 break
    #
    #         # ===== APPLY THE CHOSEN OPTION =====
    #         wait_days = best_travel_choice["wait_days"]
    #
    #         # 1) Record any waiting/stay days (advance current_day for each stay)
    #         for _ in range(wait_days):
    #             if current_day >= self.usa_map.days:
    #                 break  # safety: do not write past available days
    #             stay_record = self.meta_data["daily"][current_day]
    #             stay_record["path"] = [current_city]
    #             stay_record["distance"] = 0.0
    #             stay_record["fuel"] = 0.0
    #             stay_record["risk"] = 0.0
    #             stay_record["edges"] = []
    #             # no change to totals because nothing happened
    #             current_day += 1
    #
    #         # 2) Now record the travel that happens on the travel day (current_day)
    #         if current_day >= self.usa_map.days:
    #             # no more days left to record travel
    #             break
    #
    #         travel_day = current_day
    #         day_record = self.meta_data["daily"][travel_day]
    #         day_record["path"] = best_travel_choice["metadata"]["path"]
    #         day_record["distance"] = best_travel_choice["metadata"]["distance"]
    #         day_record["fuel"] = 0.0
    #         day_record["risk"] = 0.0
    #         day_record["edges"] = []
    #
    #         # collect edge-level info for that travel day (use travel_day when reading edge info)
    #         for (u, v) in best_travel_choice["metadata"]["edges"]:
    #             edge_info = self.usa_map.get_edge_info(u, v, travel_day)
    #             if edge_info is None:
    #                 continue
    #             day_record["fuel"] += edge_info["gallons"]
    #             # day_record["risk"] += edge_info["avg_weather"]  # optional
    #             day_record["edges"].append({
    #                 "from": u,
    #                 "to": v,
    #                 "distance": edge_info["distance"],
    #                 "gallons": edge_info["gallons"],
    #                 # "avg_weather": edge_info["avg_weather"],
    #                 "weight": edge_info["weight"]
    #             })
    #
    #             # accumulate totals
    #             self.meta_data["total"]["distance"] += edge_info["distance"]
    #             self.meta_data["total"]["fuel"] += edge_info["gallons"]
    #             # self.meta_data["total"]["risk"] += edge_info["avg_weather"]  # optional
    #
    #         # 3) Advance calendar: travel consumes one day (so move to next day)
    #         current_day = travel_day + 1
    #
    #         # 4) Move along the final_path sequence
    #         current_city = best_travel_choice["dest"]
    #         i += best_travel_choice["prefix_len"]
    #
    #     # final totals
    #     self.meta_data["total"]["days_travelled"] = current_day

    def populate_meta_data(self, final_path,day):
        """
        Travel as far as possible each day along final_path.
        Pure greedy travel: reach the farthest city possible each day.
        Dynamically calculates the day count based on travel.
        """
        remaining_cities = list(final_path)
        current_day = day
        current_city = remaining_cities[0]
        i = 1  # index of the next city in final_path

        while i < len(remaining_cities) and current_day < self.usa_map.days:

            # Get cities reachable today from current_city
            reachable = self.usa_map.reachable_within_day_min_cost(current_city, current_day)

            best_prefix_len = 0
            best_dest = None
            best_metadata = None

            # Find the farthest sequence of cities matching final_path
            for j in range(i, len(remaining_cities)):
                possible_dest = remaining_cities[j]

                if possible_dest in reachable:
                    metadata = reachable[possible_dest]
                    prefix_nodes = metadata["path"]

                    # Ensure path matches final_path exactly
                    if prefix_nodes == remaining_cities[i - 1: j + 1]:
                        if (j - i + 1) > best_prefix_len:
                            best_prefix_len = j - i + 1
                            best_dest = possible_dest
                            best_metadata = metadata

            # Fallback: travel to next city only if no longer prefix matches
            if best_dest is None:
                next_city = remaining_cities[i]
                if next_city in reachable:
                    best_prefix_len = 1
                    best_dest = next_city
                    best_metadata = reachable[next_city]
                else:
                    break  # cannot travel today → stop

            # Write travel record for this day
            day_record = self.meta_data["daily"][current_day]
            day_record["path"] = best_metadata["path"]
            day_record["distance"] = best_metadata["distance"]
            day_record["fuel"] = 0.0
            day_record["risk"] = 0.0
            day_record["edges"] = []

            # Accumulate edge info
            for (u, v) in best_metadata["edges"]:
                info = self.usa_map.get_edge_info(u, v, current_day)
                if info is None:
                    continue

                day_record["fuel"] += info["gallons"]
                day_record["risk"] += info["avg_weather"]
                day_record["edges"].append({
                    "from": u, "to": v,
                    "distance": info["distance"],
                    "gallons": info["gallons"],
                    "avg_weather": info["avg_weather"],
                    "weight": info["weight"]
                })

                self.meta_data["total"]["distance"] += info["distance"]
                self.meta_data["total"]["fuel"] += info["gallons"]
                self.meta_data["total"]["risk"] += info["avg_weather"]

            # Advance to next city/day
            current_city = best_dest
            i += best_prefix_len
            current_day += 1

        # Store total days traveled dynamically
        # print('pop',current_day)
        self.meta_data["total"]["days_travelled"] = current_day


    def populates_meta_data(self, final_path,day=0):
        """
        Travel as far as possible each day along final_path.
        No waiting logic. Pure greedy maximum-prefix-per-day travel.
        """
        remaining_cities = list(final_path)
        current_day = day
        current_city = remaining_cities[0]
        i = 1  # next required city in the path

        while i < len(remaining_cities) and current_day < self.usa_map.days:

            # get all reachable cities for today
            reachable = self.usa_map.reachable_within_day_min_cost(
                current_city, current_day
            )

            best_prefix_len = 0
            best_dest = None
            best_metadata = None

            # try to reach the farthest city in today’s reachable set
            for j in range(i, len(remaining_cities)):
                possible_dest = remaining_cities[j]

                if possible_dest in reachable:
                    metadata = reachable[possible_dest]
                    prefix_nodes = metadata["path"]

                    # must match the required final_path prefix exactly
                    if prefix_nodes == remaining_cities[i - 1: j + 1]:
                        if (j - i + 1) > best_prefix_len:
                            best_prefix_len = (j - i + 1)
                            best_dest = possible_dest
                            best_metadata = metadata

            # if nothing matched, fallback to next city only
            if best_dest is None:
                next_city = remaining_cities[i]
                if next_city in reachable:
                    best_prefix_len = 1
                    best_dest = next_city
                    best_metadata = reachable[next_city]
                else:
                    break  # cannot travel today → stop

            # write travel record for today
            day_record = self.meta_data["daily"][current_day]
            day_record["path"] = best_metadata["path"]

            day_record["distance"] = best_metadata["distance"]
            day_record["fuel"] = 0.0
            day_record["risk"] = 0.0
            day_record["edges"] = []

            # gather edge statistics
            for (u, v) in best_metadata["edges"]:
                info = self.usa_map.get_edge_info(u, v, current_day)
                if info is None:
                    continue

                day_record["fuel"] += info["gallons"]
                day_record["risk"] += info["avg_weather"]
                day_record["edges"].append({
                    "from": u, "to": v,
                    "distance": info["distance"],
                    "gallons": info["gallons"],
                    "avg_weather": info["avg_weather"],
                    "weight": info["weight"]
                })

                self.meta_data["total"]["distance"] += info["distance"]
                self.meta_data["total"]["fuel"] += info["gallons"]
                self.meta_data["total"]["risk"] += info["avg_weather"]

            # advance
            current_city = best_dest
            i += best_prefix_len
            current_day += 1
        #self.dest_reach_day+=current_day
        self.meta_data["total"]["days_travelled"] = current_day

    # def pairwise_comp(self, city_a: str, city_b: str):
    #     # Dynamic programming, memoization approach
    #     key = tuple((city_a, city_b))
    #     # if already exists
    #     if key in self.pair_cache:
    #         return self.pair_cache[key]
    #     #start_day =
    #     result = self.search(city_a, city_b)  # A* search for the given pair
    #     print(result.get("days_travelled"))
    #     if result is None:  # Cannot reach
    #         no_path = {
    #             "path": [],
    #             "cost": float("inf"),
    #             "days_travelled": None,
    #             "meta_data": None
    #         }
    #
    #         self.pair_cache[key] = no_path
    #         return no_path
    #
    #     self.pair_cache[key] = result
    #     return result
#TODO:AI
    def pairwise_comp(self, city_a: str, city_b: str):
        """
            Compute A→B cost for all possible start days.
            Store:
                result[start_day] = {cost, days_travelled, path}
        """
        key = (city_a, city_b)
        if key in self.pair_cache:
            return self.pair_cache[key]

        result_by_day = {}

        for start_day in range(self.max_days):
            res = self.search(city_a, city_b, start_day=start_day)

            if res is None:
                result_by_day[start_day] = {
                    "path": [],
                    "cost": float("inf"),
                    "days_travelled": None,
                    "meta_data": None
                }
            else:
                result_by_day[start_day] = res

        self.pair_cache[key] = result_by_day
        #print(result_by_day)
        return result_by_day
#AI
    def build_cost_matrix(self, cities):
        costs = {}

        for i, city_a in enumerate(cities):
            for city_b in cities[i + 1:]:
                # Compute multi-day A→B
                ab = self.pairwise_comp(city_a, city_b)
                costs[(city_a, city_b)] ={
                     int(day): int(ab[day]["cost"]) if ab[day]["cost"] != float("inf") else 999
                    for day in ab
                }
                 #{day: int(ab[day]["cost"]) for day in ab}

                # Compute multi-day B→A
                ba = self.pairwise_comp(city_b, city_a)
                costs[(city_b, city_a)] = {
                     int(day): int(ba[day]["cost"]) if ba[day]["cost"] != float("inf") else 999
                    for day in ba
                }


        # cost(city, city) = 0 for all days
        for city in cities:
            costs[(city, city)] = {day: 0.0 for day in range(self.max_days)}

        return costs

    # def build_cost_matrix(self, cities):  # real_distance + weather risk matrix
    #     """
    #     Build a cost directory
    #     :param cities: cities needed to travel
    #     :return: cost matrix
    #     """
    #     costs = {} #hold cost of pairs
    #     #Loop over each pair of cities
    #     for i, city_a in enumerate(cities):
    #         for city_b in cities[i + 1:]:
    #             #Compute cost of A-B
    #             result_cost_a_b = self.pairwise_comp(city_a, city_b)
    #             cost = result_cost_a_b["cost"] if result_cost_a_b is not None else float('inf')
    #             costs[(city_a, city_b)] = cost
    #             #Note cost of B-A is not same as A-B
    #             result_cost_b_a = self.pairwise_comp(city_b, city_a)
    #             cost = result_cost_b_a["cost"] if result_cost_b_a is not None else float('inf')
    #             costs[(city_b, city_a)] = cost
    #
    #     for city in cities: #cost of itself
    #         costs[(city, city)] = 0.0
    #     print(costs)
    #     return costs


    # def improve_path(self, route, costs, max_iter=200):
    #     def route_cost(r):
    #         total_route_cost = 0.0
    #         for i in range(len(r) - 1):
    #             total_route_cost += costs.get((r[i], r[i + 1]), float('inf'))
    #         return total_route_cost
    #
    #     best = route[:]
    #     best_cost = route_cost(best)
    #     improved = True
    #     it = 0
    #
    #     while improved and it < max_iter:
    #         improved = False
    #         it += 1
    #         n = len(best)
    #         for i in range(1, n - 1):
    #             for j in range(i + 1, n - 1):
    #
    #                 new = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
    #                 cost = route_cost(new)
    #                 if cost < best_cost:
    #                     best = new
    #                     best_cost = cost
    #                     improved = True
    #                     break
    #
    #             if improved:
    #                 break
    #
    #     return best

    # def execute_multiplicity(self, city_list):
    #     trip = [] #smaller trips to reach from one reqd city to another reqd city
    #     total_distance = 0.0  # total distance travelled
    #     total_days = 0 #days took to travel
    #     total_fuel = 0.0 #total fuel
    #     #total_risk = 0.0
    #     current_day = 0  #current day
    #     final_path = []
    #     schedule = [] #daily itearnary
    #
    #     for i in range(len(city_list) - 1):
    #         city_a = city_list[i]
    #         city_b = city_list[i + 1]
    #
    #         result = self.pairwise_comp(city_a, city_b)
    #         print(result)
    #         print('exc',result['cost'])
    #         if result is None or result["cost"] == float("inf"):  # cannot reach
    #             trip.append({"from": city_a, "to": city_b}) #, "ok": False
    #             continue
    #
    #         trip.append({
    #             "from": city_a, "to": city_b, #"ok": True,
    #             "cost": result["cost"],
    #             "days_travelled": result["days_travelled"],
    #             "path": result["path"],
    #             "meta_data": result["meta_data"]
    #         })
    #
    #         trip_metadata = result.get("meta_data")
    #         if trip_metadata:
    #             for day_index, day_rec in trip_metadata["daily"].items():
    #                 schedule.append({
    #                     "global_day": current_day + int(day_index),
    #                     "trip_from": city_a,
    #                     "trip_to": city_b,
    #                     "path": day_rec["path"],
    #                     "distance": day_rec["distance"],
    #                     "fuel": day_rec["fuel"],
    #                     #"risk": day_rec["risk"],
    #                     "edges": day_rec["edges"]
    #                 })
    #                 total_distance += day_rec["distance"]
    #                 total_fuel += day_rec["fuel"]
    #                 #total_risk += day_rec["risk"]
    #                 total_days += 1
    #             current_day += trip_metadata["total"]["days_travelled"]
    #
    #         if result["path"]:
    #             if not final_path:
    #                 final_path.extend(result["path"])
    #             else:
    #                 final_path.extend(result["path"][1:])
    #
    #     return {
    #         "ordered_route": city_list,
    #         "trip": trip,
    #         "total_distance": total_distance,
    #         "total_days": total_days,
    #         "total_fuel": total_fuel,
    #         "daily_route": schedule,
    #         "final_path": final_path
    #     }

    def execute_multiplicity(self, city_list):
        """
        Plan a multi-day travel route through a list of cities.

        Args:
            city_list (list): Ordered list of cities to travel through.

        Returns:
            dict: Contains ordered route, trips info, total distance, days, fuel,
                  daily itinerary, and the full path taken.
        """
        trips = []  # Info about travel between consecutive cities
        total_distance = 0.0
        total_days = 0
        total_fuel = 0.0
        current_day = 0  # Tracks the global day count
        final_path = []  # Complete path for all trips
        daily_schedule = []  # Day-wise itinerary

        for i in range(len(city_list) - 1):
            city_from = city_list[i]
            city_to = city_list[i + 1]

            # Compute pairwise travel info
            result = self.pairwise_comp(city_from, city_to)
            print(current_day,result[current_day])
            print('Cost for this trip:', result[current_day]['cost'])

            # If cities cannot be reached
            if result is None or result[current_day]["cost"] == float("inf"):
                trips.append({"from": city_from, "to": city_to})
                continue
            print(result[current_day]["days_travelled"])
            # Record the trip info
            trips.append({
                "from": city_from,
                "to": city_to,
                "cost": result[current_day]["cost"],
                "days_travelled": result[current_day]["days_travelled"],
                "path": result[current_day]["path"],
                "meta_data": result[current_day]["meta_data"]
            })

            # Process multi-day travel metadata if available
            meta = result[current_day]["meta_data"]
            if meta:
                for day_index, day_info in meta["daily"].items():
                    daily_schedule.append({
                        "global_day": current_day + int(day_index),
                        "trip_from": city_from,
                        "trip_to": city_to,
                        "path": day_info["path"],
                        "distance": day_info["distance"],
                        "fuel": day_info["fuel"],
                        "edges": day_info["edges"]
                    })
                    total_distance += day_info["distance"]
                    total_fuel += day_info["fuel"]
                    total_days += 1

                # Update the current global day after this trip
                print(meta["total"]["days_travelled"])
                current_day += meta["total"]["days_travelled"]
                print(current_day)
            # Update the final full path
            if result[current_day]["path"]:
                if not final_path:
                    final_path.extend(result[current_day]["path"])
                else:
                    # Avoid repeating the last city of previous path
                    final_path.extend(result[current_day]["path"][1:])

        return {
            "ordered_route": city_list,
            "trip": trips,
            "total_distance": total_distance,
            "total_days": total_days,
            "total_fuel": total_fuel,
            "daily_route": daily_schedule,
            "final_path": final_path
        }

    def trip(self, initial_city_list):
        # needs at least two cities to travel
        if len(initial_city_list) < 2:
            return "need least two cities to travel"

        city_list = list(dict.fromkeys(initial_city_list))  # remove duplicates if any

        # fail-safe
        for city in city_list:
            if city not in self.usa_map.cities:
                return f"city {city} not found in map"
        #Take a initial guess and jump start the A*
        costs = self.build_cost_matrix(city_list)  # Pairwise costs
        print('costmatrix',costs)  #input order is calculated
        initial_route = build_initial_route(city_list, costs)  # initial greedy path
        print('initial',initial_route)
        improved_route = improve_path(initial_route, costs)  # optimize route
        print('opti',improved_route)
        #both opt and final is same
        aggregated = self.execute_multiplicity(improved_route)
        aggregated["initial_route"] = initial_route
        aggregated["final_route"] = improved_route
        aggregated["pairwise_costs"] = costs
        print(aggregated)
        return aggregated

    def heuristic(self, current: str, next_city: str, destination: str) -> float:
        """
        method to the h function of to next city towards dest city, says how many gas will be needed
        It penalizes to move in the opposite direction of dest
        """

        # get latitude and longitude coordinates of current, dest and next city
        coordinates_current_city = self.coordinates[current]
        coordinates_next_city = self.coordinates[next_city]
        coordinates_destination_city = self.coordinates[destination]

        if coordinates_current_city and coordinates_destination_city:
            dx = coordinates_destination_city[0] - coordinates_current_city[0]
            dy = coordinates_destination_city[1] - coordinates_current_city[1]

            straight_dist = math.hypot(dx, dy)  # get distance between current and destination
            base_h = straight_dist / self.usa_map.base_mpg  # estimated fuel needed to reach goal
        else:
            base_h = 0.0

        # direction penalty disabled
        if not self.use_direction_penalty:
            return base_h

        # coordinates not found
        if not coordinates_current_city or not coordinates_next_city or not coordinates_destination_city:
            return base_h

        vx = coordinates_destination_city[0] - coordinates_current_city[0]
        vy = coordinates_destination_city[1] - coordinates_current_city[1]
        ux = coordinates_next_city[0] - coordinates_current_city[0]
        uy = coordinates_next_city[1] - coordinates_current_city[1]

        theta = math.hypot(vx, vy) * math.hypot(ux, uy)
        # reached
        if theta == 0:
            return base_h

        cos_theta = (vx * ux + vy * uy) / theta  # cos angle tells the direction

        # towards dest
        if cos_theta >= 0:
            return base_h

        # opposite direction
        penalty = (-cos_theta) * self.penalty_factor * (straight_dist / self.usa_map.base_mpg)

        return base_h + penalty  # add penalty
