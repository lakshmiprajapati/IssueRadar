
import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 120000,
});

// Check backend health
export const checkHealth = async () => {
  const response = await API.get("/health");
  return response.data;
};

// Analyze GitHub repository
export const analyzeRepository = async (repositoryUrl) => {
  const response = await API.post("/api/v1/analyze", {
    repository_url: repositoryUrl,
  });
  return response.data;
};

// Predict individual issue category
export const predictIssue = async (title, description = "") => {
  const response = await API.post("/api/v1/predict", {
    title,
    description,
  });
  return response.data;
};

export default API;
