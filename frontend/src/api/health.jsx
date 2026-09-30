import { useEffect, useState } from "react";
import api from "./axios";

export default function Health() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchHealth() {
      try {
        const response = await api.get("/health");
        setData(response.data);
      } catch (requestError) {
        console.error("Health check failed:", requestError);
        setError("Unable to reach the API.");
      }
    }

    fetchHealth();
  }, []);

  return (
    <>
      <h1>Health check status</h1>
      {error && <p>{error}</p>}
      {!data && !error && <p>Checking API health…</p>}
      {data && (
        <>
          <p>{JSON.stringify(data)}</p>
          <p>{data.status}</p>
        </>
      )}
    </>
  );
}
setError("Unable to reach the API.");
