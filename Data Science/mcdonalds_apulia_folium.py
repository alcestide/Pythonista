import pandas
import folium

places = pandas.read_csv("mcdonalds_apulia.csv",
                         encoding='utf-8')

m = folium.Map(location=[41.118707942656066,
                         16.867880656156167],
               tiles='OpenStreetMap',
               zoom_start=9)

for i, row in places.iterrows():
    lat = places.at[i,'lat']
    lng = places.at[i,'long']
    restaurant = places.at[i,'restaurant']
    popupText = (str(places.at[i,'restaurant'])
                 + '<br>' + str(places.at[i,'street'])
                 + '<br>'' '+ str(places.at[i,'city']))
    folium.Marker(location=[lat, lng],
                  popup=popupText,
                  icon=folium.Icon(color='green')).add_to(m)

m.save("mcdonalds_apulia.html")
