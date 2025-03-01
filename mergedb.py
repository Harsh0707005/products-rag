import faiss
import pandas as pd
import numpy as np
import os

faiss_files = [
    "products1_index.faiss",
    "products2_index.faiss",
    "products3_index.faiss",
    "products4_index.faiss",
    "products5_index.faiss"
]

pkl_files = [
    "products1-preprocessed.pkl",
    "products2-preprocessed.pkl",
    "products3-preprocessed.pkl",
    "products4-preprocessed.pkl",
    "products5-preprocessed.pkl"
]


merged_index = faiss.read_index(faiss_files[0])

for faiss_file in faiss_files[1:]:
    temp_index = faiss.read_index(faiss_file)
    merged_index.merge_from(temp_index, 0)  # 0 means "no checks"

faiss.write_index(merged_index, "merged_index.faiss")

merged_df = pd.concat([pd.read_pickle(pkl) for pkl in pkl_files], ignore_index=True)

merged_df.to_pickle("merged-preprocessed.pkl")