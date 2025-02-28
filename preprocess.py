import json
import pandas

with open("products.json") as f:
    products = json.load(f)

products_df = pandas.DataFrame(products)
products_df.to_pickle("preprocessed.pkl")