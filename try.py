import json

# Load the JSON file
with open("merged.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Filter out objects where "primary_image" is an empty string
filtered_data = [obj for obj in data if obj.get("primary_image")]

# Save the cleaned data back to the file
with open("merged.json", "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, indent=4)
