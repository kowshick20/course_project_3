import heapq
import math
from collections import defaultdict
from functools import lru_cache
from typing import List, Tuple

from CityGraph import CityGraph


def get_cost(costs, city_a, city_b, day):
    """
    Give the cost of travelling from city_a to city_b in a given day
    """
    entry = costs.get((city_a, city_b), {})
    if isinstance(entry, dict):
        return float(entry.get(day, float("inf")))
    return float("inf")


class Astar:
    MAX_DISTANCE = 600

    def __init__(self, graph: CityGraph, coordinates,
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
            "total": {"distance": 0.0, "fuel": 0.0, "risk": 0.0, "days_travelled": 0, "total_path": [], "mpg": 0.0}}

    # If I start today from city A what cities do I need to cover before preachings city B
    def search(self, current, destination, start_day=0):
        """
        A* start algorithm
        :param start_day: Do the search for given day
        :param current: current city
        :param destination: destination city
        :return: metadata JSON if when reached else empty
        """
        # reset
        self.meta_data = {
            "daily": defaultdict(lambda: {"path": [], "distance": 0.0, "fuel": 0.0, "risk": 0.0, "edges": []}),
            "total": {"distance": 0.0, "fuel": 0.0, "risk": 0.0, "days_travelled": 0, "total_path": [], "mpg": 0.0}}

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
                self.populate_meta_data(path, start_day)  # generate the meta_data

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
                heapq.heappush(open_pq, (new_g, new_f, dest, new_day, new_path))  # push the element

        return None  # if goal cannot be reached

    # How to cover the trip
    def populate_meta_data(self, final_path, day):
        """
        Backtracks and reconstructs the travel details,

        No wait days since you will travel to any one of the best possible node always
          greedy, chooses the max distance to travel per day
         :param final_path: Local path of reaching city a to b
         :param day: Travel day
        """
        remaining_cities = list(final_path)  # cities to travel
        current_day = day  # current day
        current_city = remaining_cities[0]  # start city
        index_of_city = 1  # Index if city
        total_travel_days = 0  # days travelled so far

        # Util visited all cities and have days left for travel
        while index_of_city < len(remaining_cities) and current_day < self.usa_map.days:

            # Cities reachable from here, on a given day
            reachable = self.usa_map.reachable_within_day_min_cost(current_city, current_day)

            # store the results
            cities_covered = 0  # store the most number of cities, index_of_city can cover today
            best_dest = None  # final destination
            best_metadata = None

            # Greedy aim for the max distance
            for index_of_next in range(index_of_city, len(remaining_cities)):
                possible_dest = remaining_cities[index_of_next]  # possible next cities

                if possible_dest in reachable:  # is it reachable today?
                    metadata = reachable[possible_dest]
                    travel_next_cities = metadata["path"]

                    # Travel to the farthest available city
                    if travel_next_cities == remaining_cities[index_of_city - 1: index_of_next + 1]:
                        if (index_of_next - index_of_city + 1) > cities_covered:
                            cities_covered = index_of_next - index_of_city + 1  # travel as much as possible
                            best_dest = possible_dest
                            best_metadata = metadata

            # fail-safe, no farthest? travel to next city
            if best_dest is None:
                next_city = remaining_cities[index_of_city]
                if next_city in reachable:
                    cities_covered = 1
                    best_dest = next_city
                    best_metadata = reachable[next_city]
                else:
                    break  # cannot travel today → stop

            # Record the travel for the day
            day_record = self.meta_data["daily"][current_day]
            day_record["path"] = best_metadata["path"]
            day_record["distance"] = best_metadata["distance"]
            day_record["fuel"] = 0.0
            day_record["risk"] = 0.0
            day_record["edges"] = []

            # Talk to the graph to get the fuel,distance details
            for (from_city, to_city) in best_metadata["edges"]:
                # get details from graph
                info = self.usa_map.get_edge_info(from_city, to_city, current_day)
                if info is None:
                    continue

                day_record["fuel"] += info["gallons"]
                day_record["risk"] += info["avg_weather"]
                total_travel_days += info['days']
                day_record["edges"].append({
                    "from": from_city, "to": to_city,
                    "distance": info["distance"],
                    "gallons": info["gallons"],
                    "avg_weather": info["avg_weather"],
                    "mpg": info["mpg"],

                })
                # rolling average of mpg
                mpg = info["mpg"]
                existing = self.meta_data["total"]["mpg"]
                avg = (existing * total_travel_days + mpg) / (total_travel_days + 1)
                self.meta_data["total"]["mpg"] = avg
                self.meta_data["total"]["distance"] += info["distance"]
                self.meta_data["total"]["fuel"] += info["gallons"]
                self.meta_data["total"]["risk"] += info["avg_weather"]

            # Continue the loop
            current_city = best_dest
            index_of_city += cities_covered
            current_day += 1

        self.meta_data["total"]["days_travelled"] = total_travel_days

    def compute_local_travel_cost(self, city_a, city_b):
        """
        Calculate the route cost for travelling from city-a to city-b for the period of 30 days
        :param city_a: start city
        :param city_b: end city
        :return: travel cost for 30 days
        """
        # if cost is already computed, return the same
        # DP approach
        key = (city_a, city_b)
        if key in self.pair_cache:
            return self.pair_cache[key]

        result_by_day = {}

        # all 30 days
        for start_day in range(self.max_days):
            if city_a != 'wait' and city_b != 'wait':
                # get the metadata and cost
                res = self.search(city_a, city_b, start_day=start_day)
            else:
                res = None

            # if no path for day, no cost
            if res is None:
                result_by_day[start_day] = {
                    "path": [],
                    "cost": float("inf"),
                    "days_travelled": None,
                    "meta_data": None
                }
            else:
                result_by_day[start_day] = res
        # memoize
        self.pair_cache[key] = result_by_day

        return result_by_day

    def build_cost_matrix(self, cities):
        """
        Build the cost matrix for travel one city to the other for the period of 30 days
        :param cities: List of cities needed to be visited
        :return: the cost matrix
        """
        costs = {}  # cost
        # calculate the cost of each pair of the city
        for i, city_a in enumerate(cities):
            for city_b in cities[i + 1:]:
                # cost of city_A to City_b
                ab = self.compute_local_travel_cost(city_a, city_b)
                costs[(city_a, city_b)] = {
                    int(day): int(ab[day]["cost"]) if ab[day]["cost"] != float("inf") else 999
                    for day in ab
                }

                # cost of city_A to City_b, not A-B and B-A are not same
                ba = self.compute_local_travel_cost(city_b, city_a)
                costs[(city_b, city_a)] = {
                    int(day): int(ba[day]["cost"]) if ba[day]["cost"] != float("inf") else 999
                    for day in ba
                }

        # Cost for the same city is always 0
        for city in cities:
            costs[(city, city)] = {day: 0.0 for day in range(self.max_days)}

        return costs

    def execute_multiplicity(self, route_list, initial_cities, costs):
        """
        Plan the day-wise travel iternary
        :param route_list: final travel route
        :param initial_cities: initial required cities to be visited
        :param costs: cost of visiting each city
        :return: day wise plan data
        """
        daily_schedule = []  # schedule for each day
        trips = []  # short trip needed
        final_path = []  # final route
        travel_days = 0
        total_days = len(route_list)  # total days required to travel
        total_distance = 0.0
        total_fuel = 0.0
        buffer_of_real_cities_stack = []  # hold the last real city
        distance_for_day = 0  # distance travelled for the day
        for city in range(len(route_list) - 1):
            # cannot travel for more than the maximum travel days
            if travel_days > self.usa_map.days:
                print("Wait amd save not possible, suggesting greedy route")
                self.build_greedy_route(initial_cities, costs)  # rebuild the travel iternary using greedy approach

            city_a = route_list[city]  # start city
            city_b = route_list[city + 1]  # end city
            # if start city is not wait, store it in buffer
            if city_a != 'wait':
                buffer_of_real_cities_stack.append(city_a)

            # If start of end city is wait, log the day as 'waiting'
            if city_a == 'wait' or city_b == 'wait':
                daily_schedule.append({
                    "global_day": travel_days,
                    "trip_from": city_a,
                    "trip_to": city_b,
                    "path": ["WAIT"],
                    "distance": 0.0,
                    "fuel": 0.0,
                    "edges": []
                })
                travel_days += 1  # move to next day

                final_path.append("WAIT")  # final path is wait
                # if we have a end city by not start city
                if buffer_of_real_cities_stack and (city_b != 'wait'):
                    # get the previous start city
                    prev_city = buffer_of_real_cities_stack.pop()
                    # compute the cost of travelling to prev_city to city_b
                    print(prev_city,city_b)
                    # get travel data for today
                    result = self.compute_local_travel_cost(prev_city, city_b)
                    meta = result[travel_days]["meta_data"]
                    if meta:  # if available, log it
                        for day_index, day_info in meta["daily"].items():
                            # compute the distance for the day
                            distance_for_day += day_info["distance"]
                            # if the distance exceeds the max distance for the day, move to next day
                            if distance_for_day > self.MAX_DISTANCE:
                                travel_days += 1
                                distance_for_day =0
                            daily_schedule.append({
                                "global_day": int(travel_days),
                                "trip_from": prev_city,
                                "trip_to": city_b,
                                "path": day_info["path"],
                                "distance": day_info["distance"],
                                "fuel": day_info["fuel"],
                                "edges": day_info["edges"]
                            })


                            total_distance += day_info["distance"]
                            total_fuel += day_info["fuel"]
                    # append to the final path
                    if not final_path:
                        final_path.extend([city_a, city_b])
                    else:
                        final_path.append(city_b)

            else:
                # do the same for if start_city and end_city exists
                result = self.compute_local_travel_cost(city_a, city_b)
                meta = result[travel_days]["meta_data"]
                if meta:
                    for day_index, day_info in meta["daily"].items():
                        distance_for_day += day_info["distance"]
                        if distance_for_day > self.MAX_DISTANCE:
                            travel_days += 1
                            distance_for_day = 0
                        daily_schedule.append({
                            "global_day": int(travel_days),
                            "trip_from": city_a,
                            "trip_to": city_b,
                            "path": day_info["path"],
                            "distance": day_info["distance"],
                            "fuel": day_info["fuel"],
                            "edges": day_info["edges"]
                        })

                        total_distance += day_info["distance"]
                        total_fuel += day_info["fuel"]

                if not final_path:
                    final_path.extend([city_a, city_b])
                else:
                    final_path.append(city_b)

        # return the overall travel data
        return {
            "ordered_route": route_list,
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
        # Take a initial guess and jump start the A*
        costs = self.build_cost_matrix(city_list)  # Pairwise costs

        travel_route = self.build_cost_aware_route(city_list, costs)  # initial greedy path

        # both opt and final is same
        aggregated = self.execute_multiplicity(travel_route, city_list, costs)  # improved_route
        aggregated["final_route"] = travel_route
        aggregated["pairwise_costs"] = costs

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
            straight_dist = 0.0
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

    def get_travel_days(self, city_a, city_b):
        """
        Days requires to travel from one coty to another
        :param city_a: start city
        :param city_b: goal city
        :return: Days required
        """
        # A->B take the same total number of days distance and route might vary
        info = self.usa_map.get_edge_info(city_a, city_b, 0)
        if info is None:
            return 0
        return info['days']

    def build_greedy_route(self, cities, costs, max_days=30):
        """
        Initial route identifier using greed algo, that takes in the travel constraints
        :param cities: Cities to visit
        :param costs: Cost of travelling for the next 30 days
        :param max_days: Can travel for 30 day at max
        :return: the initial greedy route
        """

        # pick the city with the least cost as the starting city
        best_city = min(cities, key=lambda c: sum(get_cost(costs, c, o, 1) for o in cities if o != c))
        route = [best_city]
        unvisited = set(cities) - {best_city}

        current_day = 1  # starting on day 1

        # build the complete route
        while unvisited and current_day <= max_days:
            current_city = route[-1]

            # pick the next city that has the minimum cost as of today
            next_city = min(
                unvisited,
                key=lambda c: get_cost(costs, current_city, c, current_day)
            )

            travel_cost = get_cost(costs, current_city, next_city, current_day)

            travel_days = self.get_travel_days(current_city, next_city)
            # If infinite cost cannot travel there today so try tomorrow
            if travel_cost == float("inf"):
                current_day += 1
                if current_day > max_days:
                    break
                continue

            # Note the successfully travel location
            route.append(next_city)
            unvisited.remove(next_city)
            # Increment day
            current_day += travel_days

        return route

    def build_cost_aware_route(self, cities, costs):
        """
        The goal is to travel to all the cities within the max_days , minimizing  the total cost
        as much as possible
        :param cities: cities to visit
        :param costs: Cost of visiting each city
        :return: final day aware route and cost minimized as much as possible
        """

        def get_cost_on_day(start_city, end_city, today):
            return get_cost(costs, start_city, end_city, today)

        no_of_cities = len(cities) - 1  # number of cities to visit

        # Hold days required for travel one city to another, for all given cities
        travel_days = []

        # get the travel days required for travelling from one city to another, pairwise
        for i in range(no_of_cities):
            city_a, city_b = cities[i], cities[i + 1]
            days = int(self.get_travel_days(city_a, city_b))
            if days < 1:
                days = 1
            travel_days.append(days)

        # the days needed to travel to complete the entire trip, starting from today
        remaining_days = [0] * (no_of_cities + 1)
        for k in range(no_of_cities - 1, -1, -1):
            remaining_days[k] = remaining_days[k + 1] + travel_days[k]

        # latest day to start from the given city, to successfully complete the entire travel
        latest_start = [self.usa_map.days - remaining_days[k] + 1 for k in range(no_of_cities)]

        @lru_cache(None)  # momoization
        def dp(index, given_day):
            """
            starting from kth city on the given day, calculate the minimum cost
             needed to travel to the last city
            :param index: city index
            :param given_day: travel day
            :return: the total cost
            """
            # last city
            if index == no_of_cities:
                return 0.0

            # travel beyond the maximum given days is not possible
            if given_day > self.usa_map.days:
                return float("inf")

            # possible start at the latest possible start day at max
            start_day = latest_start[index]
            if start_day < given_day:
                return float("inf")

            best = float("inf")  # initilize the best travel cost
            city_start, city_end = cities[index], cities[index + 1]  # get the cities
            days_needed = travel_days[index]  # travel from city to another

            # check for the least cost of travel starting from given day to latest possible day
            for possible_start in range(given_day, start_day + 1):
                cost_today = get_cost_on_day(city_start, city_end, possible_start)  # get cost of today
                if cost_today == float("inf"):
                    continue
                next_trip = possible_start + days_needed  # possible_start this trip today, then when will I possible_start the next trip
                # Next trip starts at given day, what is the cost of travelling to the end
                subsequent_travel = dp(index + 1, next_trip)  # recursive call
                if subsequent_travel == float("inf"):
                    continue
                # total cost for completing, if started today
                total_cost_for_today = cost_today + subsequent_travel
                best = min(best, total_cost_for_today)  # replace best

            return best

        # Total cost of starting from city 0 on day 1
        total_cost = dp(0, 1)
        if total_cost == float("inf"):
            raise Exception("No feasible schedule found")

        # Determine the starting day for each city
        start_days = [None] * no_of_cities  # best day to travel
        city_index = 0  # trip details
        earliest_day = 1  # earliest to start
        while city_index < no_of_cities:  # until valid city
            # start city and stop city
            source, destination = cities[city_index], cities[city_index + 1]
            travel = travel_days[city_index]
            # initilize
            best_start = None
            best_val = float("inf")
            last_start = latest_start[city_index]  # last possible

            # Decide on the start day, from each city
            for start in range(earliest_day, last_start + 1):
                cost = get_cost_on_day(source, destination, start)
                if cost == float("inf"):
                    continue
                next_day = start + travel
                subsequent = dp(city_index + 1, next_day)
                if subsequent == float("inf"):
                    continue
                total = cost + subsequent
                if total < best_val:
                    best_val = total
                    best_start = start
            if best_start is None:
                raise Exception("Reconstruction failed")
            start_days[city_index] = best_start  # finalize the best possible_start
            # increment loop
            earliest_day = best_start + travel
            city_index += 1

        daily = []  # daily itenary
        current_day = 1  # day
        # all cities
        for k in range(no_of_cities):
            start_date = start_days[k]
            begin_city = cities[k]
            travel_needed = travel_days[k]
            # Any buffer days, then wait on those
            while current_day < start_date:
                daily.append('wait')
                current_day += 1

            for dd in range(travel_needed):
                # day label
                daily.append(begin_city)

                current_day += 1

        return daily


