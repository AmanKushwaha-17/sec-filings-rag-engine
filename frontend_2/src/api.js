// src/api.js
// Abstraction layer for interacting with the FastAPI backend.

const API_BASE = "http://localhost:8000";

export const api = {
  async getHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (!res.ok) throw new Error("Network response was not ok");
      return await res.json();
    } catch (error) {
      console.error("Error fetching health:", error);
      return { status: "error" };
    }
  },

  async getCompanies() {
    try {
      const res = await fetch(`${API_BASE}/api/companies`);
      if (!res.ok) throw new Error("Network response was not ok");
      return await res.json();
    } catch (error) {
      console.error("Error fetching companies:", error);
      return [];
    }
  },

  async sendChat(query) {
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      
      const data = await res.json();
      
      // Handle the new APIErrorResponse structure
      if (!res.ok) {
        throw new Error(data.message || `API Error: ${res.status}`);
      }
      return data;
    } catch (error) {
      console.error("Chat Error:", error);
      throw error;
    }
  }
};
