import csv
import random
from datetime import date, timedelta
import pandas as pd
from PathFinder import PathFinder
from CityGraph import CityGraph
from map import plot_route

BASE_MPG = 45.0
MAX_MILES_PER_DAY = 8 * 75  # 600 miles/day
PENALTY_FACTOR = 10.0  # used if direction penalty is enabled
MAX_DAYS = 30
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
    "Savannah",
    "Knoxville",
    "Columbia",
    "Atlanta",
    "Jacksonville",
    "Miami",
    "Orlando",
    "Detroit",
    "Indianapolis"
]


def collect_weather_data(weather_csv):
    weather_df = pd.read_csv(weather_csv)
    weather_df = weather_df.sort_values(['city'])
    return {
        city: group['condition'].tolist() for city,group in weather_df.groupby("city")
    }



def load_city_graph_from_csv(csv_path,weather_csv):
    cities = {}
    edges_base = []
    coordinates = {}
    # Fixed weather map
    weather_risk_map = {"sun": 1,"cloudy":1 , "rain": 5, "snow": 10}

    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract city names
            from_city = row["from_city"]
            to_city = row["to_city"]

            # Extract fields
            distance = float(row["real_distance_miles"])
            elevation_from = float(row["from_elevation_miles"])
            elevation_to = float(row["to_elevation_miles"])
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
                "mpg": float(row["mpg"]),
                "gallons": float(row["gallons_needed"]),
                "days": days
            })

    # Create a 30-day weather forecast for each city
    weather = collect_weather_data(weather_csv)

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
    print(f"-> Trip from: {start_date} to {end_date} ")

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
                  f"({edge['distance']:.2f} mi, "
                  f"{edge['gallons']:.2f} gal), ")

    plot_route(route, report, coordinates)  # print the interactive USA map


def calender_date(day):
    """
    Util function,converting day to date
    :param day: day of month
    :return: date in nov 25
    """
    start = date(2025, 11, 1)
    return start + timedelta(day)


def pick_random_cities(n):
    """
    Test Util, pick N City at random from the list
    :return: random city list
    """
    shuffled = cities_list[:]  # copy
    random.shuffle(shuffled)  # scramble order
    return shuffled[:n]  # return N random cities


if __name__ == "__main__":
    cities, edges_base, weather, weather_risk_map, coordinates = load_city_graph_from_csv("travel_routes_final.csv",
                                                                                          "weather_november_2025 (1).csv")
    # print(coordinates)
    cg = CityGraph(cities, edges_base, weather,
                   weather_risk_map,BASE_MPG,MAX_MILES_PER_DAY,MAX_DAYS,)

    multi_city = PathFinder(cg, coordinates, PENALTY_FACTOR, True)

    N = 15
    travel_list = pick_random_cities(N)

    result = multi_city.trip(travel_list)

    print(print_full_metadata(result, coordinates))
