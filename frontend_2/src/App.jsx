import React, { useState, useEffect } from 'react';
import Sidebar from './Sidebar';
import ChatInterface from './ChatInterface';
import EvidenceDrawer from './EvidenceDrawer';
import { api } from './api';
import './App.css';

function App() {
  const [isReady, setIsReady] = useState(false);
  const [healthStatus, setHealthStatus] = useState("Checking backend status...");
  const [companies, setCompanies] = useState([]);
  
  const [selectedSource, setSelectedSource] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

  // Poll backend health until ready
  useEffect(() => {
    let interval;
    const checkHealth = async () => {
      const data = await api.getHealth();
      if (data.status === "API ready") {
        setIsReady(true);
        setHealthStatus("Ready");
        clearInterval(interval);
        
        // Fetch companies once ready
        const comps = await api.getCompanies();
        setCompanies(comps);
      } else {
        setHealthStatus(data.status || "Waiting for backend...");
      }
    };

    checkHealth();
    if (!isReady) {
      interval = setInterval(checkHealth, 3000);
    }
    
    return () => clearInterval(interval);
  }, [isReady]);

  const handleCitationClick = (source) => {
    setSelectedSource(source);
    setIsDrawerOpen(true);
  };

  const handleDrawerClose = () => {
    setIsDrawerOpen(false);
    // Optional: delay clearing source to allow animation to finish
    setTimeout(() => setSelectedSource(null), 400);
  };

  if (!isReady) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner"></div>
        <div style={{ color: "var(--text-secondary)" }}>{healthStatus}</div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <Sidebar companies={companies} />
      
      <div className={`main-content ${isDrawerOpen ? 'drawer-open' : ''}`}>
        <ChatInterface 
          onCitationClick={handleCitationClick} 
          sendMessage={api.sendChat}
          isDrawerOpen={isDrawerOpen}
        />
      </div>
      
      <EvidenceDrawer 
        isOpen={isDrawerOpen} 
        onClose={handleDrawerClose} 
        source={selectedSource} 
      />
    </div>
  );
}

export default App;
