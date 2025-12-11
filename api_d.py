import json
import requests

# ---- STEP 1: Download GeoJSON from ArcGIS ----

url = "https://services5.arcgis.com/GfwWNkhOj9bNBqoJ/arcgis/rest/services/NYC_Neighborhood_Tabulation_Areas_2020/FeatureServer/0/query?where=1=1&outFields=*&outSR=4326&f=pgeojson"

print("Downloading NTA GeoJSON...")
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"Failed to download GeoJSON: {response.status_code}")

geojson_data = response.json()

# ---- STEP 2: Convert to lightweight JSON ----

lite = []
for feature in geojson_data["features"]:
    props = feature["properties"]
    geom  = feature["geometry"]

    nta_name = (
        props.get("NTAName") or 
        props.get("NTACode") or 
        props.get("Name")
    )

    lite.append({
        "NTAName": nta_name,
        "geometry": geom
    })

# ---- STEP 3: Save output ----

out_path = "nta_polygons.json"
with open(out_path, "w") as f:
    json.dump(lite, f)

print("🎉 Saved:", out_path)
print("Total NTAs:", len(lite))
