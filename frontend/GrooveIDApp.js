
import React, { useState } from 'react';

export default function GrooveIDApp() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', file);
    try {
      const res = await fetch('/api/identify', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error('Identification failed.');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Groove ID</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!file || loading}>
        {loading ? 'Identifying…' : 'Identify Record'}
      </button>
      {result && (
        <div>
          <h2>{result.title}</h2>
          <p>Artist: {result.artist}</p>
          <p>Label: {result.label}</p>
          <p>Cat#: {result.catalog_number}</p>
          <p>Confidence: {Math.round(result.confidence * 100)}%</p>
          <a href={result.discogs_url} target="_blank">View on Discogs</a>
        </div>
      )}
      {error && <p style={{color: 'red'}}>{error}</p>}
    </div>
  );
}
