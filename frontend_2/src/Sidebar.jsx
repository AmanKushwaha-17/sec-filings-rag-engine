import React from 'react';
import { Building2 } from 'lucide-react';
import './Sidebar.css';

export default function Sidebar({ companies }) {
  return (
    <div className="sidebar glass-panel">
      <div className="sidebar-header">
        <Building2 size={24} color="var(--accent-color)" />
        <h2>SEC Intelligence</h2>
      </div>
      
      <div className="sidebar-list">
        <div className="list-title">INDEXED COMPANIES</div>
        {companies.length === 0 ? (
          <div className="empty-state">Loading companies...</div>
        ) : (
          companies.map((c) => (
            <div key={c.ticker} className="company-item">
              <span className="ticker">{c.ticker}</span>
              <span className="name" title={c.company_name}>{c.company_name}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
