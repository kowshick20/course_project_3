# MultiCityPlanner.py
# Requires your existing CityGraph and AStarPlanner classes to be available in the same namespace.
# Usage:
#   planner = MultiCityPlanner(city_graph, astar_planner)
#   result = planner.plan_unordered_trip(["A","B","C","D","E"])
#   result contains final_order, total_cost, per_leg_results, telemetry

import itertools
import math
import copy
from typing import List, Dict, Any, Tuple

class MultiCityPlanner:
    def __init__(self, city_graph, astar_planner, verbose: bool = False):
        """
        city_graph: an instance of CityGraph
        astar_planner: an instance of AStarPlanner (must implement search(start, goal) -> dict or None)
        """
        self.graph = city_graph
        self.astar = astar_planner
        self.verbose = verbose
        # cache: (a,b) -> search result dict (path, cost, day_reached, telemetry)
        self.pair_cache: Dict[Tuple[str,str], Dict[str,Any]] = {}

    # ---------------------------
    # Pairwise A* evaluation & caching
    # ---------------------------
    def pairwise_search(self, a: str, b: str) -> Dict[str,Any]:
        key = tuple(sorted((a,b)))
        if key in self.pair_cache:
            return self.pair_cache[key]
        # Run A* from a -> b
        res = self.astar.search(a, b)
        # If no path found, set a very large cost and an "empty" telemetry
        if res is None:
            fake = {"path": [], "cost": float("inf"), "day_reached": None, "telemetry": None}
            self.pair_cache[key] = fake
            return fake
        # store result keyed by ordered (a,b) for clarity too
        self.pair_cache[key] = res
        return res

    # ---------------------------
    # Build pairwise cost matrix for the requested city list
    # ---------------------------
    def build_cost_matrix(self, cities: List[str]) -> Dict[Tuple[str,str], float]:
        costs = {}
        for (i, a) in enumerate(cities):
            for b in cities[i+1:]:
                res = self.pairwise_search(a, b)
                cost = res["cost"] if res is not None else float("inf")
                costs[(a,b)] = cost
                costs[(b,a)] = cost
                if self.verbose:
                    print(f"pair cost {a}->{b} = {cost}")
        # ensure diagonal zero
        for c in cities:
            costs[(c,c)] = 0.0
        return costs

    # ---------------------------
    # Greedy nearest neighbor on pairwise A* costs (returns ordered route list)
    # ---------------------------
    def greedy_route(self, cities: List[str], costs: Dict[Tuple[str,str], float]) -> List[str]:
        # choose starting city: pick city with smallest sum of costs to others
        avg_costs = []
        for c in cities:
            s = sum(costs.get((c,o), float("inf")) for o in cities if o != c)
            avg_costs.append((s, c))
        avg_costs.sort()
        start = avg_costs[0][1]
        route = [start]
        unvisited = set(cities) - {start}
        while unvisited:
            last = route[-1]
            # pick unvisited with smallest cost from last
            next_city = min(unvisited, key=lambda x: costs.get((last,x), float("inf")))
            route.append(next_city)
            unvisited.remove(next_city)
        return route

    # ---------------------------
    # 2-opt local improvement (swap segments) to reduce route cost
    # ---------------------------
    def two_opt(self, route: List[str], costs: Dict[Tuple[str,str], float], max_iter=1000) -> List[str]:
        best = route[:]
        improved = True
        it = 0
        def route_cost(r):
            s = 0.0
            for i in range(len(r)-1):
                s += costs.get((r[i], r[i+1]), float("inf"))
            return s
        best_cost = route_cost(best)
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(best)
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    if j - i == 1:  # adjacent edges, continue
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_cost = route_cost(new_route)
                    if new_cost + 1e-9 < best_cost:
                        best = new_route
                        best_cost = new_cost
                        improved = True
                        if self.verbose:
                            print(f"2-opt improved cost to {best_cost:.3f} at iter {it}")
                        break
                if improved:
                    break
        return best

    # ---------------------------
    # Take ordered route and execute legs (run A* per leg), aggregate telemetry
    # ---------------------------
    def execute_route(self, ordered: List[str]) -> Dict[str,Any]:
        """
        ordered: list of cities in visitation order (e.g. ["C","A","D","B"...])
        returns dict with per-leg results and aggregated telemetry
        """
        legs = []
        total_distance = 0.0
        total_fuel = 0.0
        total_risk = 0.0
        full_path_nodes = []
        day_offset = 0  # sum of days used so far; next leg's day indexing depends on A* implementation (here A* uses day-layer starting at 0)
        detailed_daily_schedule = []  # list of day records across the whole multi-leg trip

        for i in range(len(ordered)-1):
            a = ordered[i]
            b = ordered[i+1]
            # run A* for this leg (note: this will itself assume day=0..D-1 in A*; we do not offset the weather days here,
            # but you could implement day_offset shifting if you want the weather to be relative across legs)
            res = self.pairwise_search(a, b)
            if res is None or res["cost"] == float("inf"):
                # cannot reach leg
                leg_result = {"from": a, "to": b, "ok": False, "reason": "unreachable"}
                legs.append(leg_result)
                if self.verbose:
                    print(f"Leg {a}->{b} unreachable")
                continue

            # res contains: path, cost, day_reached, telemetry (per-day info for that leg)
            legs.append({
                "from": a,
                "to": b,
                "ok": True,
                "cost": res["cost"],
                "day_reached": res.get("day_reached"),
                "path": res.get("path"),
                "telemetry": res.get("telemetry")
            })

            # aggregate telemetry: note telemetry is per-leg; we append day records sequentially
            leg_tel = res.get("telemetry")
            if leg_tel:
                # per-day
                for d_idx, drec in leg_tel["daily"].items():
                    # day index in global schedule = day_offset + int(d_idx)
                    day_record = {
                        "global_day": day_offset + int(d_idx),
                        "leg_from": a,
                        "leg_to": b,
                        "path": drec["path"],
                        "distance": drec["distance"],
                        "fuel": drec["fuel"],
                        "risk": drec["risk"],
                        "edges": drec["edges"]
                    }
                    detailed_daily_schedule.append(day_record)
                    total_distance += drec["distance"]
                    total_fuel += drec["fuel"]
                    total_risk += drec["risk"]
                # update day_offset by days used in leg
                day_offset += leg_tel["total"]["days_used"] if leg_tel["total"].get("days_used") else 0

            # append nodes to global path
            if res.get("path"):
                if not full_path_nodes:
                    full_path_nodes.extend(res["path"])
                else:
                    # avoid duplicate node at boundary
                    full_path_nodes.extend(res["path"][1:])

        aggregated = {
            "ordered_route": ordered,
            "legs": legs,
            "total_distance": total_distance,
            "total_fuel": total_fuel,
            "total_risk": total_risk,
            "detailed_daily_schedule": detailed_daily_schedule,
            "full_path_nodes": full_path_nodes
        }
        return aggregated

    # ---------------------------
    # Public method: plan_unordered_trip
    # ---------------------------
    def plan_unordered_trip(self, city_list: List[str]) -> Dict[str,Any]:
        """
        Input: travel_list list of city names the user wants to visit.
        Output:
          - final_order: list of cities in visitation order
          - result: aggregated telemetry + per-leg details
        """
        if len(city_list) < 2:
            return {"error": "need at least two cities for a trip"}

        # Make set unique and validate cities in graph
        city_list = list(dict.fromkeys(city_list))  # preserve order, remove duplicates
        for c in city_list:
            if c not in self.graph.cities:
                return {"error": f"city {c} not in graph"}

        # 1) Build pairwise costs (A* for each pair) - O(n^2) A* calls
        costs = self.build_cost_matrix(city_list)

        # 2) Greedy initial route
        initial = self.greedy_route(city_list, costs)
        if self.verbose:
            print("Greedy initial route:", initial)

        # 3) 2-opt improvement
        improved = self.two_opt(initial, costs)
        if self.verbose:
            print("Improved route:", improved)

        # 4) Execute final route leg-by-leg (call A* results again from cache)
        agg = self.execute_route(improved)
        agg["initial_route"] = initial
        agg["final_route"] = improved
        agg["pairwise_costs"] = costs
        return agg



