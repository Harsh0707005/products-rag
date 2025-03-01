import json


with open("merged.json", "r", encoding="utf-8") as file:
    data = json.load(file)

filtered_data = [obj for obj in data if obj.get("primary_image")]

with open("merged.json", "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, indent=4)
