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
            self.index_file = f"{products_file[:-5]}_index.faiss"
        else:
            self.products_file = None
            self.preprocessed_file = None
            self.index_file = None

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
        products_df['embedding'] = text_embeddings.tolist()

        print("Generating Image Embeddings")
    
        products_df["image_embeddings"] = products_df.apply(
            lambda row: self.generate_image_embeddings(row["primary_image"]) 
            if row.get("primary_image") 
            else np.zeros((self.img_embedding_dim,)), 
            axis=1
        )
        
        print("Combining embeddings")

        products_df["combined_embeddings"] = products_df.apply(
            lambda row: np.concatenate((row["embedding"], row["image_embeddings"])), 
            axis=1
        )

        embeddings_array = np.array(products_df["combined_embeddings"].tolist()).astype("float32")

        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        faiss.write_index(index, f"{self.products_file[:-5]}_index.faiss")
        self.index_file = f"{self.products_file[:-5]}_index.faiss"

    def search_products(self, query, index=None, k=15):
        print("Searching products with text")
        if not (self.index_file and os.path.isfile(self.index_file)):
            print("FAISS index file not found")
            return
        elif not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocess file not found")
            return

        query_embedding = self.model.encode([query])
        zero_img_embedding = np.zeros((1, self.img_embedding_dim))
        combined_query_embedding = np.hstack([query_embedding, zero_img_embedding]).astype('float32')
        index = faiss.read_index(self.index_file)
        distances, indices = index.search(combined_query_embedding, k * 2)

        results = []
        seen_product_names = set()
        products_df = pandas.read_pickle(self.preprocessed_file)

        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            product = products_df.iloc[idx].to_dict()
            product["similarity"] = float(1 / (1 + distance))

            if product["product_name"] not in seen_product_names:
                seen_product_names.add(product["product_name"])
                results.append(product)

            if len(results) >= k:
                break

        return results

    def search_with_image(self, image_path, k=15):
        print("Searching products with image")
        if not (self.index_file and os.path.isfile(self.index_file)):
            print("FAISS index file not found")
            return
        elif not (self.preprocessed_file and os.path.isfile(self.preprocessed_file)):
            print("Preprocess file not found")
            return

        image_embedding = self.generate_image_embeddings(image_path, path=True)
        zero_text_embedding = np.zeros(self.text_embedding_dim)
        combined_embedding = np.concatenate([zero_text_embedding, image_embedding])
        combined_embedding = np.reshape(combined_embedding, (1, -1)).astype('float32')

        index = faiss.read_index(self.index_file)
        distances, indices = index.search(combined_embedding, k * 2)

        results = []
        seen_product_names = set()
        products_df = pandas.read_pickle(self.preprocessed_file)

        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]

            product = products_df.iloc[idx].to_dict()
            product["similarity"] = float(1 / (1 + distance))

            if product["product_name"] not in seen_product_names and product["product_url"] != "":
                seen_product_names.add(product["product_name"])
                results.append(product)

            if len(results) >= k:
                break

        return results
        

if __name__ == "__main__":

    rag = RAGModel("aelia.json")

    rag.preprocess()
    rag.generate_embeddings()

    results = rag.search_products("watches")
    with open("results.json", "w") as f:
        json.dump(results, f, indent = 4)

    # results = rag.search_with_image("search_image.jpg")

    # with open("results.json", "w") as f:
    #     json.dump(results, f, indent=4)