import folium
from folium.plugins import PolyLineTextPath

def plot_route(route, report, city_coordinates, file_name="route_map.html"):
    """
    Util function to visualize the route details in the USA map
    :param route: trip route
    :param report: trip details
    :param city_coordinates: location of city in map
    :param file_name: file-name to store
    """
    # Make the USA map
    usa_center = [39.50, -98.35]
    m = folium.Map(location=usa_center, zoom_start=5)
    for city in route:
        if city != 'wait':
            start_city = city  # initial start city
            break
    end_city = route[-1]  # end city

    # Mark each required city to travel
    for city in route:
        if city == 'wait':
            continue
        lat, lon = city_coordinates[city]  # get coordinates of a city
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{city}</b>",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # Connect the cities based on the final path
    route_coordinates = [(city_coordinates[c][0], city_coordinates[c][1]) for c in route if c!='wait']
    # design the route marker
    polyline = folium.PolyLine(
        locations=route_coordinates,
        weight=5,
        opacity=0.8,
        color="red"
    ).add_to(m)
    # add direction symbol
    PolyLineTextPath(
        polyline,
        text="     >     ",
        repeat=True,
        offset=7,
        attributes={"font-size": "24px", "fill": "blue"}
    ).add_to(m)
    # mark start city specially
    # identify the start city using coordinates
    start_lat, start_lon = city_coordinates[start_city]
    # mark it with a flag symbol
    folium.Marker(
        location=[start_lat, start_lon],
        popup=f"<b>START: {start_city}</b>",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    # mark end city specially
    # identify the end city using coordinates
    end_lat, end_lon = city_coordinates[end_city]
    # mark it with stop icon
    folium.Marker(
        location=[end_lat, end_lon],
        popup=f"<b>END: {end_city}</b>",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)



    # Save the html file
    m.save(file_name)

    print(f"\n Route map saved to: {file_name}")
