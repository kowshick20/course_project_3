import csv
import random
from datetime import date, timedelta

from ASTAR2 import Astar
from CityGraph import CityGraph
from map import plot_route_folium

BASE_MPG = 45.0
MAX_MILES_PER_DAY = 8 * 75  # 600 miles/day
PENALTY_FACTOR = 10.0  # used if direction penalty is enabled

cities_list = [
    "Portland",
    "Manchester",
    "Burlington",
    "Boston",
    "New York City",
    "Albany",
    "Buffalo",
    "Providence",
    "Hartford",
    "Philadelphia",
    "Pittsburgh",
    "Newark",
    "Wilmington",
    "Baltimore",
    "Charleston",
    "Columbus",
    "Cleveland",
    "Richmond",
    "Charlotte",
    "Raleigh",
    "Louisville",
    "Nashville",
    "Memphis",
    "Columbia",
    "Atlanta",
    "Jacksonville",
    "Miami",
    "Orlando",
    "Detroit",
    "Indianapolis"
]


def load_city_graph_from_csv(csv_path):
    cities = {}
    edges_base = []
    coordinates = {}
    # Fixed weather map
    weather_risk_map = {"sunny": 1, "rain": 5, "snow": 10}

    # Weather types to pick from
    weather_types = ["sunny", "rain", "snow"]

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract city names
            from_city = row["from_city"]
            to_city = row["to_city"]

            # Extract fields
            distance = float(row["real_distance_miles"])
            elevation_from = int(row["from_elevation_ft"])
            elevation_to = int(row["to_elevation_ft"])
            latitude_1 = float(row["from_lat"])
            longitude_1 = float(row["from_lon"])
            latitude_2 = float(row["to_lat"])
            longitude_2 = float(row["to_lon"])
            days = float(row["travel_days"])

            # Build cities dict
            if from_city not in cities:
                cities[from_city] = {"elevation": elevation_from}
                coordinates[from_city] = (latitude_1, longitude_1)

            if to_city not in cities:
                cities[to_city] = {"elevation": elevation_to}
                coordinates[to_city] = (latitude_2, longitude_2)

            # Build edges_base entry
            edges_base.append({
                "from": from_city,
                "to": to_city,
                "distance": distance,
                # you CAN include these if your A* uses them:
                "mpg": float(row["mpg"]),
                "gallons": float(row["gallons_needed"]),
                "risk": float(row["avg_risk_factor"]),
                "days": days
            })

    # Create a stub 5-day weather forecast for each city
    weather = {
        city: [random.choice(weather_types) for _ in range(30)]
        for city in cities
    }

    return cities, edges_base, weather, weather_risk_map, coordinates


def print_full_metadata(report, coordinates):
    """
    Output function, to print the trip details
    Draw the interactive USA map to visualize the output
    :param report: Metadata of travel
    :param coordinates: Coordinates of every city
    """
    # Final route
    route = report.get("final_route", [])
    print("\nFinal Route:")
    print("   " + " → ".join(route))

    #Overall summary
    distance = report.get('total_distance', 0)
    fuel = report.get('total_fuel', 0)
    if fuel==0:
        mpg = 0
    else:
        mpg = distance / fuel
    start_date = '2025-11-01'  # constant
    end_date = calender_date(report.get('total_days', 0))
    print("\nSummary:")
    print(f"-> Total Distance: {distance:.2f} miles")
    print(f"-> Total Fuel: {fuel:.2f} gallons")
    print(f"-> Average miles per Gallon: {mpg:.2f} ")
    print(f"-> Trip from: {start_date} to {end_date} gallons")
    print(f"-> Total Trips: {len(report.get('trip', []))}")

    # Per-trip breakdown
    print("\nTrip Details:")
    for idx, trip in enumerate(report.get("trip", []), start=1):
        print(f"\nTrip {idx}: {trip['from']} → {trip['to']}")
        print(f"\tCost: {trip.get('cost', 0):.2f}")
        print(f"\tDays: {trip.get('days_travelled', 0)}")
        print(f"\tPath: {' → '.join(trip.get('path', []))}")

        metadata = trip.get("meta_data", {})
        daily_info = metadata.get("daily", {})

        for day_number, day_idx in enumerate(sorted(daily_info), start=1):
            day = daily_info[day_idx]
            print(f"\tDay {day_number}:")
            print(f"\t Cities: {' → '.join(day.get('path', []))}")
            print(f"\t Distance: {day.get('distance', 0):.2f} mi")
            print(f"\t Fuel: {day.get('fuel', 0):.2f} gal")

            print("\tEdges:")
            for edge in day.get("edges", []):
                print(f"\t\t\t{edge['from']} → {edge['to']} "
                      f"({edge['distance']:.2f} mi, "
                      f"{edge['gallons']:.2f} gal, "
                      f"mpg={edge['mpg']:.2f})")

    # Day-by-day schedule
    print("\nDaily Schedule:")
    for entry in report.get("daily_route", []):
        if entry['trip_from'] == 'wait' or entry['trip_to'] == 'wait':
            print(f"\nDate {calender_date(entry['global_day'])}: wait")
        else:
            print(f"\nDate {calender_date(entry['global_day'])}: {entry['trip_from']} → {entry['trip_to']}")
        print(f"\tPath: {' → '.join(entry['path'])}")
        print(f"\tDistance: {entry['distance']:.2f} mi")
        print(f"\tFuel: {entry['fuel']:.2f} gallon")
        print("\tEdges:")
        for edge in entry.get("edges", []):
            print(f"\t\t{edge['from']} → {edge['to']} "
                  f"\t\t({edge['distance']:.2f} mi, "
                  f"\t\t{edge['gallons']:.2f} gal, ")

    plot_route_folium(route, report, coordinates)  # print the interactive USA map


def calender_date(day):
    """
    Util function,converting day to date
    :param day: day of month
    :return: date in nov 25
    """
    start = date(2025, 11, 1)
    return start + timedelta(days=day)


def pick_random_cities(n):
    """
    Test Util, pick N City at random from the list
    :return: random city list
    """
    shuffled = cities_list[:]  # copy
    random.shuffle(shuffled)  # scramble order
    return shuffled[:n]  # return N random cities


if __name__ == "__main__":
    cities, edges_base, weather, weather_risk_map, coordinates = load_city_graph_from_csv("travel_data.csv")
    # print(coordinates)
    cg = CityGraph(cities, edges_base, weather,
                   weather_risk_map, BASE_MPG,
                   MAX_MILES_PER_DAY)

    multi_city = Astar(cg, coordinates, PENALTY_FACTOR, True)

    N = 15
    travel_list = pick_random_cities(N)
    print(travel_list)

    # travel_list = [
    #     "Portland",
    #     "Manchester",
    #     "Burlington",
    #     "Boston",
    #     "New York City",
    #     "Albany",
    #     "Buffalo",
    #     "Jacksonville",
    #     "Hartford",
    #     "Philadelphia"]

    result = multi_city.trip(travel_list)

    print(print_full_metadata(result, coordinates))
