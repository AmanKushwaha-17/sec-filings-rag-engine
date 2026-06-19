import React, { useState, useRef, useEffect } from 'react';
import { Send, Search, Loader2 } from 'lucide-react';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import './ChatInterface.css';

export default function ChatInterface({ onCitationClick, sendMessage, isDrawerOpen }) {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const userQuery = query;
    setQuery("");
    setMessages(prev => [...prev, { role: "user", content: userQuery }]);
    setIsLoading(true);

    try {
      const response = await sendMessage(userQuery);
      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: response.answer,
        sources: response.sources,
        analyzer: response.analyzer_info
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: "error", 
        content: error.message || "Failed to generate response." 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderMarkdownWithCitations = (content, sources) => {
    let html = DOMPurify.sanitize(marked.parse(content));
    
    return (
      <div 
        className="markdown-body"
        dangerouslySetInnerHTML={{ __html: html }}
        onClick={(e) => {
          const btn = e.target.closest('.citation-chip');
          if (btn) {
            const index = parseInt(btn.dataset.id, 10) - 1;
            if (sources && sources[index]) {
              onCitationClick(sources[index]);
            }
          }
        }}
      />
    );
  };

  const processContent = (content) => {
    return content.replace(/\[(\d+)(?:,\s*[^\]]+)?\]/g, (match, id) => {
      return `<button class="citation-chip" data-id="${id}">${match}</button>`;
    });
  };

  return (
    <div className={`chat-interface glass-panel`}>
      <div className="messages-area">
        {messages.length === 0 && (
          <div className="welcome-state">
            <Search size={48} color="var(--accent-color)" opacity={0.5} />
            <h2>What would you like to know?</h2>
            <p>Ask about revenue, supply chain risks, or business strategy across SEC filings.</p>
          </div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} className={`message-wrapper ${msg.role}`}>
            <div className={`message ${msg.role}`}>
              {msg.role === 'assistant' && msg.analyzer?.metadata_filtering_applied && (
                <div className="analyzer-chip">
                  Filtered to: {msg.analyzer.detected_ticker}
                </div>
              )}
              
              {msg.role === 'user' ? (
                <div className="user-text">{msg.content}</div>
              ) : msg.role === 'error' ? (
                <div className="error-text">{msg.content}</div>
              ) : (
                renderMarkdownWithCitations(processContent(msg.content), msg.sources)
              )}
              
              {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                <div className="sources-list">
                  <span className="sources-label">Sources:</span>
                  {msg.sources.map((s, i) => (
                    <button 
                      key={i} 
                      className="source-tag"
                      onClick={() => onCitationClick(s)}
                    >
                      [{i+1}] {s.ticker}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message-wrapper assistant">
            <div className="message loading">
              <Loader2 className="spinner" size={20} />
              <span>Analyzing filings & generating response...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about SEC filings..."
            disabled={isLoading}
          />
          <button type="submit" disabled={!query.trim() || isLoading} className="submit-btn">
            <Send size={20} />
          </button>
        </form>
      </div>
    </div>
  );
}
