from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from model import RAGModel

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

os.makedirs("uploads", exist_ok=True)

rag = RAGModel("merged.json")
# rag.preprocess()
# rag.generate_embeddings()

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = rag.search_products(query)
    return jsonify(results)

@app.route("/search_image", methods=["POST"])
def search_image():
    image = request.files.get("image")
    if not image:
        return jsonify({"error": "Image is required"}), 400

    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    results = rag.search_with_image(image_path)

    os.remove(image_path)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
