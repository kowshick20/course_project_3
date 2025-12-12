import folium
from folium.plugins import PolyLineTextPath

def plot_route_folium(route, report, city_coordinates, file_name="route_map.html"):
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

    # add route details as pop-up
    for idx, trip in enumerate(report["trip"], start=1):

        city_from = trip["from"]
        city_to = trip["to"]
        if city_to == 'wait' or city_from == 'wait':
            continue

        # locate the city's using coordinates
        lat1, lon1 = city_coordinates[city_from]
        lat2, lon2 = city_coordinates[city_to]
        # mid-point of route to insert the pop-up
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
#<!-- Status: {"OK" if trip.get('ok') else "Failed"}<br> -->
        # build contents of pop-up
        popup_html = f"""
        <b>Trip {idx}: {city_from} → {city_to}</b><br>
       
        Cost: {trip.get('cost', 0):.2f}<br>
        Days: {trip.get('days_travelled', 0)}<br>
        Path: {" → ".join(trip.get('path', []))}<br><br>
        <b>Daily Breakdown:</b><br>
        """

        # get the daily travel metadata
        metadata = trip.get("meta_data", {})
        daily_info = metadata.get("daily", {})

        # add the pop-up details
        for day_idx in sorted(daily_info):
            day = daily_info[day_idx]
            popup_html += f"""
            <u>Day {day_idx + 1}</u><br>
            Cities: {" → ".join(day.get("path", []))}<br>
            Distance: {day.get("distance", 0):.2f} mi<br>
            Fuel: {day.get("fuel", 0):.2f} gal<br>
            Risk: {day.get("risk", 0):.2f}<br><br>
            """

        # Add the pop-up and its marker
        folium.Marker(
            location=[mid_lat, mid_lon],
            popup=folium.Popup(popup_html, max_width=400),
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(m)

    # Save the html file
    m.save(file_name)

    print(f"\n Route map saved to: {file_name}")
