import json
import pandas
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os.path

class RAGModel():

    def __init__(self, products_file=None):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if (products_file):
            self.products_file = products_file
            self.preprocessed_file = f"{products_file[:-5]}-preprocessed.pkl"
            self.index_file = f"{products_file[:-5]}_index.faiss"
        else:
            self.products_file = None
            self.preprocessed_file = None
            self.index_file = None

    def preprocess(self, products_file=None):
        self.products_file = products_file if not self.products_file else self.products_file
        if not (self.products_file and os.path.isfile(self.products_file)):
            print("Product file does not exist")
            return
        
        with open(self.products_file) as f:
            products = json.load(f)

        products_df = pandas.DataFrame(products).fillna("")
        products_df.to_pickle(f"{self.products_file[:-5]}-preprocessed.pkl")
        self.preprocessed_file = f"{self.products_file[:-5]}-preprocessed.pkl"

    def generate_embeddings(self):
        if not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocessed file not found")
            return

        products_df = pandas.read_pickle(self.preprocessed_file)

        products_df['embedding_text'] = products_df.apply(lambda row: f"{row['brand_name']} {row['product_name']} {row['description']}",axis=1)

        embeddings = self.model.encode(products_df["embedding_text"].tolist())

        products_df['embedding'] = embeddings.tolist()

        embeddings_array = np.array(embeddings).astype("float32")

        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        faiss.write_index(index, f"{self.products_file[:-5]}_index.faiss")
        self.index_file = f"{self.products_file[:-5]}_index.faiss"

    def search_products(self, query, index=None, k=5):
        if not (self.index_file and os.path.isfile(self.index_file)):
            print("FAISS index file not found")
            return
        elif not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocess file not found")
            return
        
        query_embedding = self.model.encode([query])
        index = faiss.read_index(self.index_file)
        distances, indices = index.search(query_embedding, k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            products_df = pandas.read_pickle(self.preprocessed_file)
            product = products_df.iloc[idx].to_dict()
            product["similarity"] = float(1/(1+distance))
            results.append(product)
        return results

rag = RAGModel("products.json")

rag.preprocess()
rag.generate_embeddings()

results = rag.search_products("watches")
with open("results.json", "w") as f:
    json.dump(results, f, indent = 4)