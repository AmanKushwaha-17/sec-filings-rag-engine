import React from 'react';
import { X, FileText, Calendar, Percent } from 'lucide-react';
import './EvidenceDrawer.css';

export default function EvidenceDrawer({ isOpen, onClose, source }) {
  if (!isOpen || !source) {
    return (
      <div className="evidence-drawer glass-panel">
        <div className="empty-drawer">Select a citation to view evidence</div>
      </div>
    );
  }

  return (
    <div className={`evidence-drawer glass-panel ${isOpen ? 'open' : ''}`}>
      <div className="drawer-header">
        <div className="drawer-title">
          <FileText size={18} color="var(--accent-color)" />
          <h3>Source Evidence</h3>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close">
          <X size={20} />
        </button>
      </div>

      <div className="drawer-metadata">
        <div className="meta-item">
          <span className="meta-label">Company</span>
          <span className="meta-value highlight">{source.ticker}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Section</span>
          <span className="meta-value">{source.section}</span>
        </div>
        <div className="meta-item">
          <Calendar size={14} color="var(--text-secondary)" />
          <span className="meta-value">{source.filing_date}</span>
        </div>
        <div className="meta-item">
          <Percent size={14} color="var(--text-secondary)" />
          <span className="meta-value">Relevance: {source.rerank_score}</span>
        </div>
      </div>

      <div className="drawer-content">
        <div className="raw-text">
          {source.text}
        </div>
      </div>
    </div>
  );
}
