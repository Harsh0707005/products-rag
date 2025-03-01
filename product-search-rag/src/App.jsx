import React, { useState } from "react";
import axios from "axios";

export default function App() {
  const [query, setQuery] = useState("");
  const [image, setImage] = useState(null);
  const [results, setResults] = useState([]);

  const handleTextSearch = async () => {
    try {
      const response = await axios.get(`http://127.0.0.1:5000/search?query=${query}`);
      console.log(response.data)
      setResults(response.data);
    } catch (error) {
      console.error("Error searching by text:", error);
    }
  };

  const handleImageSearch = async () => {
    if (!image) return;
    
    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await axios.post("http://127.0.0.1:5000/search_image", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResults(response.data);
    } catch (error) {
      console.error("Error searching by image:", error);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-gray-100">
      <h1 className="text-2xl font-bold mb-4">Product Search</h1>

      <div className="mb-4">
        <input 
          type="text" 
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          placeholder="Search by text..."
          className="border p-2 rounded mr-2"
        />
        <button onClick={handleTextSearch} className="bg-blue-500 text-white px-4 py-2 rounded">
          Search
        </button>
      </div>

      <div className="mb-4">
        <input type="file" onChange={(e) => setImage(e.target.files[0])} />
        <button onClick={handleImageSearch} className="bg-green-500 text-white px-4 py-2 rounded mt-2">
          Search by Image
        </button>
      </div>

      <div className="mt-4 w-full max-w-3xl">
        {results.length > 0 ? (
          results.map((product, index) => (
            <div key={index} className="border p-4 bg-white rounded shadow mb-2">
              <h2 className="font-bold">{product.product_name}</h2>
              <p>{product.description}</p>
              {product.primary_image && <img src={product.primary_image} alt="Product" className="w-32 h-32 object-cover mt-2" />}
              <p className="text-gray-600">Similarity: {(product.similarity * 100).toFixed(2)}%</p>
            </div>
          ))
        ) : (
          <p>No results found.</p>
        )}
      </div>
    </div>
  );
}
