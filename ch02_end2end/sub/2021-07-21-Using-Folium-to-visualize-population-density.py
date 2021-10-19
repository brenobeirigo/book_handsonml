

# %%
import folium
import housing_util as hu

df = hu.load_housing_data()

df
# %%

def add_circles(point, map, min_pop, max_pop):
    # https://leafletjs.com/reference-1.6.0.html#circle
    folium.Circle(
        radius = (point.population - min_pop)/(max_pop-min_pop)*30,
        weight = 1,
        opacity = 0.4,
        location = [point.latitude, point.longitude],
        color="crimson"
    ).add_to(map)

map = folium.Map(width=600, height=400, zoom_start=2)


# Use df.apply(axis=1) to "iterate" through every row in your dataframe
df.apply(
    add_circles,
    args=(
        map, 
        min(df.population),
        max(df.population)
    ),
    axis=1
)

# Zoom in the plotted points
map.fit_bounds(map.get_bounds())

# Save the map to an HTML file
map.save('html_map_output/housing_scatter.html')

map
# %%
