import { useState } from "react";
import axios from "axios";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [image, setImage] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!query) return;
    setLoading(true);
    try {
      const { data } = await axios.get(`http://127.0.0.1:5000/search?query=${query}`);
      setResults(data);
    } catch (error) {
      console.error("Search error:", error);
    }
    setLoading(false);
  };

  const handleImageSearch = async () => {
    if (!image) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);
    try {
      const { data } = await axios.post("http://127.0.0.1:5000/search_image", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(data);
    } catch (error) {
      console.error("Image search error:", error);
    }
    setLoading(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "100vh", backgroundColor: "#f4f4f4", padding: "20px", width: "100vw" }}>
      <h1 style={{ fontSize: "24px", fontWeight: "bold", marginBottom: "16px", color: "black" }}>Product Search</h1>
      <div style={{ display: "flex", gap: "10px", marginBottom: "16px", width: "100%", maxWidth: "400px" }}>
        <input 
          type="text" 
          placeholder="Enter search query..." 
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          style={{ flex: 1, padding: "10px", borderRadius: "5px", border: "1px solid #ccc" }}
        />
        <button onClick={handleSearch} disabled={loading} style={{ padding: "10px", borderRadius: "5px", border: "none", backgroundColor: "blue", color: "white", cursor: "pointer" }}>Search</button>
      </div>
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "10px" }}>
        <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} style={{ display: "none" }} id="file-upload" />
        <label htmlFor="file-upload" style={{ display: "flex", alignItems: "center", gap: "5px", cursor: "pointer", backgroundColor: "blue", color: "white", padding: "10px", borderRadius: "5px" }}>
          Upload Image
        </label>
        {image && <button onClick={handleImageSearch} disabled={loading} style={{ padding: "10px", borderRadius: "5px", border: "none", backgroundColor: "green", color: "white", cursor: "pointer" }}>Search by Image</button>}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "16px", marginTop: "24px" }}>
        {results.map((result, index) => (
          <div 
            key={index} 
            style={{ width: "200px", padding: "10px", borderRadius: "8px", backgroundColor: "white", boxShadow: "0px 2px 10px rgba(0,0,0,0.1)", cursor: "pointer" }}
            onClick={() => window.open(result.product_url, "_blank")}
            onMouseEnter={(e) => e.currentTarget.style.opacity = "0.8"}
            onMouseLeave={(e) => e.currentTarget.style.opacity = "1"}
          >
            <img src={result.primary_image} alt={result.name} style={{ width: "100%", objectFit: "contain", borderRadius: "5px" }} />
            <h2 style={{ marginTop: "8px", fontSize: "16px", fontWeight: "bold" }}>{result.name}</h2>
            <p style={{ fontSize: "14px", color: "gray" }}>{result.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}