import json
import pandas
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os.path
import requests
from PIL import Image
from io import BytesIO

class RAGModel():

    def __init__(self, products_file=None):
        print("Initializing models")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.imgModel = SentenceTransformer("clip-ViT-B-32")

        self.text_embedding_dim = self.model.get_sentence_embedding_dimension()
        self.img_embedding_dim = self.imgModel.get_sentence_embedding_dimension()

        if self.img_embedding_dim is None:
            self.img_embedding_dim = 512

        if (products_file):
            self.products_file = products_file
            self.preprocessed_file = f"{products_file[:-5]}-preprocessed.pkl"
            self.text_index_file = f"{products_file[:-5]}_text_index.faiss"
            self.image_index_file = f"{products_file[:-5]}_image_index.faiss"
        else:
            self.products_file = None
            self.preprocessed_file = None
            self.text_index_file = None
            self.image_index_file = None

    def preprocess(self, products_file=None):
        print("Preprocessing")
        self.products_file = products_file if not self.products_file else self.products_file
        if not (self.products_file and os.path.isfile(self.products_file)):
            print("Product file does not exist")
            return
        
        with open(self.products_file) as f:
            products = json.load(f)

        products_df = pandas.DataFrame(products).fillna("")
        products_df.to_pickle(f"{self.products_file[:-5]}-preprocessed.pkl")
        self.preprocessed_file = f"{self.products_file[:-5]}-preprocessed.pkl"

    def generate_image_embeddings(self, image_loc, path=False):
        try:
            print("Processing: ", image_loc)
            if (not path):
                response = requests.get(image_loc)
                img_content = response.content
            else:
                with open(image_loc, "rb") as fb:
                    img_content = fb.read()

            img = Image.open(BytesIO(img_content)).convert("RGB")

            image_embedding = self.imgModel.encode(img)

            return image_embedding
        except Exception as e:
            print(e)
            return np.zeros((self.img_embedding_dim,))

    def generate_embeddings(self):
        print("Generating Embeddings")
        if not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocessed file not found")
            return

        products_df = pandas.read_pickle(self.preprocessed_file)

        products_df['embedding_text'] = products_df.apply(lambda row: f"{row['brand_name']} {row['product_name']} {row['description']}",axis=1)

        print("Generating Text Embeddings")
        text_embeddings = self.model.encode(products_df["embedding_text"].tolist())
        products_df['text_embedding'] = text_embeddings.tolist()
        
        text_embeddings_array = np.array(products_df["text_embedding"].tolist()).astype("float32")
        text_dimension = text_embeddings_array.shape[1]
        text_index = faiss.IndexFlatL2(text_dimension)
        text_index.add(text_embeddings_array)
        faiss.write_index(text_index, f"{self.products_file[:-5]}_text_index.faiss")
        self.text_index_file = f"{self.products_file[:-5]}_text_index.faiss"
        
        print("Generating Image Embeddings")
        products_df["image_embedding"] = products_df.apply(
            lambda row: self.generate_image_embeddings(row["primary_image"]) 
            if row.get("primary_image") 
            else np.zeros((self.img_embedding_dim,)), 
            axis=1
        )
        
        image_embeddings_array = np.array(products_df["image_embedding"].tolist()).astype("float32")
        image_dimension = image_embeddings_array.shape[1]
        image_index = faiss.IndexFlatL2(image_dimension)
        image_index.add(image_embeddings_array)
        faiss.write_index(image_index, f"{self.products_file[:-5]}_image_index.faiss")
        self.image_index_file = f"{self.products_file[:-5]}_image_index.faiss"
        
        products_df.to_pickle(self.preprocessed_file)

    def search_products(self, query, k=15):
        print("Searching products with text")
        if not (self.text_index_file and os.path.isfile(self.text_index_file)):
            print("Text index file not found")
            return
        elif not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocess file not found")
            return

        query_embedding = self.model.encode([query]).astype('float32')

        text_index = faiss.read_index(self.text_index_file)
        distances, indices = text_index.search(query_embedding, k * 2)

        results = []
        seen_product_names = set()
        products_df = pandas.read_pickle(self.preprocessed_file)

        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            product = products_df.iloc[idx].to_dict()
            product["similarity"] = float(1 / (1 + distance))

            for key, value in product.items():
                if isinstance(value, np.ndarray):
                    product[key] = value.tolist()

            if product["product_name"] not in seen_product_names:
                seen_product_names.add(product["product_name"])
                results.append(product)

            if len(results) >= k:
                break

        return results

    def search_with_image(self, image_path, k=15):
        print("Searching products with image...")
        if not (self.image_index_file and os.path.isfile(self.image_index_file)):
            print("Image index file not found")
            return
        elif not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocess file not found")
            return

        image_embedding = self.generate_image_embeddings(image_path, path=True)
        image_embedding = np.reshape(image_embedding, (1, -1)).astype('float32')

        image_index = faiss.read_index(self.image_index_file)
        distances, indices = image_index.search(image_embedding, k * 2)

        results = []
        seen_product_names = set()
        products_df = pandas.read_pickle(self.preprocessed_file)

        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            product = products_df.iloc[idx].to_dict()
            product["similarity"] = float(1 / (1 + distance))

            for key, value in product.items():
                if isinstance(value, np.ndarray):
                    product[key] = value.tolist()

            if product["product_name"] not in seen_product_names and product["product_url"] != "":
                seen_product_names.add(product["product_name"])
                results.append(product)

            if len(results) >= k:
                break

        return results
    
if __name__ == "__main__":

    rag = RAGModel("merged.json")

    # rag.preprocess()
    # rag.generate_embeddings()

    # results = rag.search_products("watches")
    # with open("results.json", "w") as f:
    #     json.dump(results, f, indent = 4)

    results = rag.search_with_image("search.jpeg")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)