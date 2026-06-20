import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Group as PanelGroup, Panel, Separator as PanelResizeHandle } from 'react-resizable-panels'
import './App.css'
import './app-additions.css'

const HISTORY_STORAGE_KEY = 'sec-research-terminal-history'
const ACTIVE_ENTRY_STORAGE_KEY = 'sec-research-terminal-active-id'
const SELECTED_SOURCE_STORAGE_KEY = 'sec-research-terminal-selected-source-id'

const SUGGESTED_QUERIES = [
  'Which semiconductor companies disclosed the greatest supply chain risk in their latest 10-K?',
  'Compare NVDA and AVGO on AI-driven demand and capacity constraints.',
  'What does TXN say about inventory normalization and end-market weakness?',
  'Summarize management discussion themes across analog semiconductor companies.',
]

const RESEARCH_THEMES = [
  'Supply chain risk',
  'AI demand',
  'Manufacturing capacity',
  'Inventory correction',
  'China exposure',
  'Capital intensity',
]

const LOADING_STEPS = [
  'Analyzing query',
  'Retrieving filings',
  'Ranking evidence',
  'Generating answer',
]

function loadStoredHistory() {
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function createEntryId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID()
  return `entry-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
}

import { formatSection, formatDate, formatFilingDate, formatSectionLabel } from './utils';

function getNormalizedScore(score) {
  const minLogit = -4
  const maxLogit = 6
  const clamped = Math.max(minLogit, Math.min(maxLogit, score))
  const ratio = (clamped - minLogit) / (maxLogit - minLogit)
  const val = 0.4 + ratio * 0.56
  return parseFloat(val.toFixed(2))
}

function extractCitationIds(label) {
  return (label.match(/\d+/g) || []).map(Number)
}

function buildAnswerSegments(answer) {
  const pattern = /\[[^\]]+\]/g
  const segments = []
  let cursor = 0
  for (const match of answer.matchAll(pattern)) {
    const index = match.index ?? 0
    if (index > cursor) segments.push({ type: 'text', value: answer.slice(cursor, index) })
    segments.push({ type: 'citation', value: match[0], ids: extractCitationIds(match[0]) })
    cursor = index + match[0].length
  }
  if (cursor < answer.length) segments.push({ type: 'text', value: answer.slice(cursor) })
  return segments.length ? segments : [{ type: 'text', value: answer }]
}

async function readResponsePayload(response) {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) return response.json()
  const text = await response.text()
  return text ? { message: text } : {}
}

export default function App() {
  const navigate = useNavigate()
  const [history, setHistory] = useState(() => loadStoredHistory())
  const [activeEntryId, setActiveEntryId] = useState(() => {
    const saved = window.sessionStorage.getItem(ACTIVE_ENTRY_STORAGE_KEY)
    if (saved === 'null') return null
    if (saved) return saved
    const initialHistory = loadStoredHistory()
    return initialHistory[0]?.id ?? null
  })
  const [selectedSourceId, setSelectedSourceId] = useState(() => {
    const saved = window.sessionStorage.getItem(SELECTED_SOURCE_STORAGE_KEY)
    if (saved === 'null') return null
    if (saved) return saved
    const initialHistory = loadStoredHistory()
    return initialHistory[0]?.sources?.[0]?.id ?? null
  })
  const [query, setQuery] = useState('')
  const [pendingQuery, setPendingQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(LOADING_STEPS[0])
  const [error, setError] = useState('')
  const [healthStatus, setHealthStatus] = useState('Checking API')
  const [companies, setCompanies] = useState([])
  const [mobileTab, setMobileTab] = useState('answer')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [tickerFilter, setTickerFilter] = useState('')

  useEffect(() => {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history))
  }, [history])

  useEffect(() => {
    if (activeEntryId === null) {
      window.sessionStorage.setItem(ACTIVE_ENTRY_STORAGE_KEY, 'null')
    } else {
      window.sessionStorage.setItem(ACTIVE_ENTRY_STORAGE_KEY, activeEntryId)
    }
  }, [activeEntryId])

  useEffect(() => {
    if (selectedSourceId === null) {
      window.sessionStorage.setItem(SELECTED_SOURCE_STORAGE_KEY, 'null')
    } else {
      window.sessionStorage.setItem(SELECTED_SOURCE_STORAGE_KEY, selectedSourceId)
    }
  }, [selectedSourceId])

  useEffect(() => {
    let cancelled = false
    async function loadBootstrapData() {
      try {
        const [healthResponse, companiesResponse] = await Promise.allSettled([
          fetch('/api/health'),
          fetch('/api/companies'),
        ])
        if (!cancelled) {
          if (healthResponse.status === 'fulfilled' && healthResponse.value.ok) {
            const payload = await readResponsePayload(healthResponse.value)
            setHealthStatus(payload.status || 'API ready')
          } else {
            setHealthStatus('API unavailable')
          }
          if (companiesResponse.status === 'fulfilled' && companiesResponse.value.ok) {
            const payload = await readResponsePayload(companiesResponse.value)
            setCompanies(Array.isArray(payload) ? payload : [])
          }
        }
      } catch {
        if (!cancelled) setHealthStatus('API unavailable')
      }
    }
    loadBootstrapData()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!isLoading) { setLoadingStep(LOADING_STEPS[0]); return }
    let index = 0
    const interval = window.setInterval(() => {
      index = (index + 1) % LOADING_STEPS.length
      setLoadingStep(LOADING_STEPS[index])
    }, 1200)
    return () => window.clearInterval(interval)
  }, [isLoading])

  useEffect(() => {
    if (!selectedSourceId) return
    document.getElementById(`source-${selectedSourceId}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [selectedSourceId, activeEntryId])

  const activeEntry = history.find(e => e.id === activeEntryId) || null
  const activeSources = activeEntry?.sources || []
  const answerSegments = buildAnswerSegments(activeEntry?.answer || '')

  const filteredCompanies = companies.filter(co =>
    co.ticker.toLowerCase().includes(tickerFilter.toLowerCase()) ||
    co.company_name.toLowerCase().includes(tickerFilter.toLowerCase())
  )

  function beginFreshResearch() {
    setActiveEntryId(null)
    setSelectedSourceId(null)
    setPendingQuery('')
    setError('')
    setMobileTab('answer')
  }

  function handleHistorySelect(entry) {
    setActiveEntryId(entry.id)
    setSelectedSourceId(entry.sources?.[0]?.id ?? null)
    setPendingQuery('')
    setError('')
    setMobileTab('answer')
  }

  function handleDeleteHistoryEntry(entryId) {
    setHistory(current => {
      const nextHistory = current.filter(e => e.id !== entryId);
      if (entryId === activeEntryId) {
        if (nextHistory.length > 0) {
          setActiveEntryId(nextHistory[0].id);
          setSelectedSourceId(nextHistory[0].sources?.[0]?.id ?? null);
        } else {
          setActiveEntryId(null);
          setSelectedSourceId(null);
        }
      }
      return nextHistory;
    });
  }

  function handleCitationClick(ids) {
    if (!ids.length) return
    setSelectedSourceId(ids[0])
    if (window.matchMedia('(max-width: 1100px)').matches) setMobileTab('sources')
  }

  function injectQuery(text) {
    setQuery(text)
    setMobileTab('answer')
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const trimmedQuery = query.trim()
    if (!trimmedQuery || isLoading) return
    setIsLoading(true)
    setError('')
    setPendingQuery(trimmedQuery)
    setMobileTab('answer')
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmedQuery }),
      })
      const payload = await readResponsePayload(response)
      if (!response.ok) throw new Error(payload.message || payload.detail || 'The research API did not return a usable response.')
      const nextEntry = {
        id: createEntryId(),
        query: trimmedQuery,
        answer: payload.answer,
        sources: payload.sources || [],
        analyzerInfo: payload.analyzer_info || null,
        executionTimeSeconds: payload.execution_time_seconds || 0,
        createdAt: new Date().toISOString(),
      }
      setHistory(current => [nextEntry, ...current])
      setActiveEntryId(nextEntry.id)
      setSelectedSourceId(nextEntry.sources?.[0]?.id ?? null)
      setHealthStatus('API ready')
      setQuery('')
      setPendingQuery('')
    } catch (requestError) {
      setError(requestError.message || 'Unable to reach the research API.')
      setHealthStatus('API unavailable')
    } finally {
      setIsLoading(false)
    }
  }

  const statusTone = healthStatus.toLowerCase().includes('ready') ? 'is-ready' : 'is-warning'

  return (
    <div className="app-shell">
      {!sidebarCollapsed ? (
        <aside className="sidebar">
          {/* Header Row */}
          <div className="corpus-header-row">
            <div className="corpus-header">
              <span className="corpus-eyebrow">Terminal</span>
              <h2 className="corpus-title">SEC Intelligence</h2>
            </div>
            <button
              type="button"
              className="ghost-icon-button collapse-toggle"
              onClick={() => setSidebarCollapsed(true)}
              aria-label="Collapse sidebar"
            >
              &lt;&lt;
            </button>
          </div>

          {/* Section 1: Recent Queries */}
          <div className={`sidebar-section is-recent ${history.length > 0 ? 'has-history' : ''}`}>
            <div className="section-heading-row">
              <span className="section-label">Recent Queries</span>
              {history.length > 0 && <span className="section-meta">{history.length}</span>}
            </div>
            
            <div className="sidebar-history-scroll">
              {history.length === 0 ? (
                <div className="sidebar-empty-state">
                  No recent searches.
                </div>
              ) : (
                history.map(entry => {
                  const isActive = entry.id === activeEntryId;
                  return (
                    <div key={entry.id} className="history-item-container">
                      <button
                        type="button"
                        className={`history-item${isActive ? ' is-active' : ''}`}
                        onClick={() => handleHistorySelect(entry)}
                      >
                        <span className="history-item-title">
                          {entry.query}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="delete-history-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteHistoryEntry(entry.id);
                        }}
                        title="Delete report"
                      >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 6h18" />
                          <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                          <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                        </svg>
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          <div className="sidebar-divider" />

          {/* Section 2: Filing Corpus */}
          <div className={`sidebar-section is-corpus ${history.length > 0 ? 'has-history' : ''}`}>
            <div className="section-heading-row">
              <span className="section-label">Filing Corpus</span>
              <span className="section-meta">{companies.length || 33}</span>
            </div>
            <div className="corpus-search-container">
              <input
                type="text"
                className="corpus-search-input"
                placeholder="Filter tickers..."
                value={tickerFilter}
                onChange={e => setTickerFilter(e.target.value)}
              />
            </div>
            <div className="corpus-list-scroll">
              {filteredCompanies.map(co => (
                <button
                  key={co.ticker}
                  type="button"
                  className="corpus-item"
                  onClick={() => navigate(`/company/${co.ticker}`)}
                  title={`View ${co.company_name} 10-K filing`}
                >
                  <span className="corpus-ticker">{co.ticker}</span>
                  <span className="corpus-name">{co.company_name}</span>
                </button>
              ))}
            </div>
          </div>
        </aside>
      ) : (
        <div className="collapsed-sidebar-stub">
          <button
            type="button"
            className="ghost-icon-button collapse-toggle"
            onClick={() => setSidebarCollapsed(false)}
            aria-label="Expand sidebar"
          >
            &gt;&gt;
          </button>
          <button type="button" className="primary-button new-research" onClick={beginFreshResearch}>
            +
          </button>
        </div>
      )}

      <div className="workspace-panel">
      <div className="workspace-shell">
        <header className="topbar">
          <div>
            <p className="eyebrow">Semiconductor Filing Intelligence</p>
            <h2>10-K RAG Intelligence</h2>
          </div>
          <div className="topbar-meta">
            <button type="button" className="primary-button new-research-topbar" onClick={beginFreshResearch}>
              New Research
            </button>
            <div className={`status-badge ${statusTone}`}>
              <span className="status-dot" />
              {healthStatus}
            </div>
            <div className="corpus-badge">{companies.length || 33} companies</div>
          </div>
        </header>

        <PanelGroup direction="horizontal" className="workspace-grid">
          <Panel defaultSize={76.47} minSize={40}>
          <main className={`answer-pane${mobileTab === 'answer' ? ' is-mobile-active' : ''}`}>
            <div className="answer-scroll-region">
              {!activeEntry && !pendingQuery && !isLoading ? (
                <section className="empty-state-card">
                  <h3>10-K RAG Intelligence</h3>
                  <p className="sq-label">Try asking</p>
                  <div className="suggested-query-list">
                    {SUGGESTED_QUERIES.map(suggestion => (
                      <button
                        key={suggestion}
                        type="button"
                        className="suggested-query-item"
                        onClick={() => injectQuery(suggestion)}
                      >
                        <svg className="sq-arrow" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 13L13 3M13 3H6M13 3v7" />
                        </svg>
                        {suggestion}
                      </button>
                    ))}
                  </div>

                </section>
              ) : (
                <div className="thread-view">
                  {/* ── User query bubble ── */}
                  <div className="query-bubble">
                    <span className="query-bubble-label">You</span>
                    <p className="query-bubble-text">{activeEntry?.query || pendingQuery}</p>
                  </div>

                  {isLoading ? (
                    /* ── Loading state below the bubble ── */
                    <div className="thread-loading">
                      <div className="thread-loading-dots">
                        <span /><span /><span />
                      </div>
                      <div className="thread-loading-info">
                        <span className="thread-loading-step">{loadingStep}</span>
                      </div>
                    </div>
                  ) : error ? (
                    <div className="error-state">
                      <p className="section-label">Request failed</p>
                      <h4>Unable to complete this filing search.</h4>
                      <p>{error}</p>
                    </div>
                  ) : (
                    /* ── AI answer section ── */
                    <section className="answer-section">
                      <div className="answer-section-header">
                        <span className="answer-icon">
                          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M8 1v14M1 8h14" opacity="0.4"/>
                            <circle cx="8" cy="8" r="3"/>
                          </svg>
                        </span>
                        <span className="answer-section-label">Answer</span>
                      </div>

                      <div className="answer-body" role="article">
                        {answerSegments.map((seg, i) =>
                          seg.type === 'text' ? (
                            <span key={i}>{seg.value}</span>
                          ) : (
                            <button
                              key={i}
                              type="button"
                              className={`citation-chip${seg.ids.includes(selectedSourceId) ? ' is-active' : ''}`}
                              onClick={() => handleCitationClick(seg.ids)}
                            >
                              {seg.value}
                            </button>
                          )
                        )}
                      </div>

                      {activeSources.length > 0 && (
                        <div className="answer-source-tickers">
                          {[...new Set(activeSources.map(s => s.ticker))].map(ticker => (
                            <button
                              key={ticker}
                              type="button"
                              className="source-ticker-chip"
                              onClick={() => navigate(`/company/${ticker}`)}
                              title={`Read full ${ticker} 10-K`}
                            >
                              {ticker} ↗
                            </button>
                          ))}
                        </div>
                      )}

                      <div className="answer-footer-note">
                        Evidence ranked by hybrid retriever · Click any citation to jump to source
                        {activeEntry && (
                          <span className="answer-footer-meta">
                            <span className="meta-pill is-time">{activeEntry.executionTimeSeconds}s</span>
                            <span className="meta-pill is-sources">{activeEntry.sources.length} sources</span>
                            {(activeEntry.analyzerInfo?.detected_tickers?.length > 0 || activeEntry.analyzerInfo?.detected_ticker) && (
                              <span className="meta-pill accent-pill">
                                {activeEntry.analyzerInfo.detected_tickers?.join(', ') || activeEntry.analyzerInfo.detected_ticker}
                              </span>
                            )}
                          </span>
                        )}
                      </div>
                    </section>
                  )}
                </div>
              )}
            </div>

            <form className="composer-shell" onSubmit={handleSubmit}>
              <label className="composer-label" htmlFor="research-query">
                Ask across the semiconductor filing corpus
              </label>
              <div className="composer-input-wrapper">
                <textarea
                  id="research-query"
                  className="composer-input"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  placeholder="Compare inventory, capex, risk factors, demand signals, or management commentary across filings..."
                  rows={3}
                />
                <button type="submit" className="composer-submit-btn" disabled={isLoading || !query.trim()}>
                  {isLoading ? 'Working...' : 'Run Research'}
                </button>
              </div>
            </form>
          </main>
          </Panel>

          <PanelResizeHandle className="resize-handle inner-handle" />

          <Panel defaultSize={23.53} minSize={10}>
          <aside className={`sources-pane${mobileTab === 'sources' ? ' is-mobile-active' : ''}`}>
            <div className="panel-header">
              <div>
                <p className="section-label">Evidence</p>
                <h3>Source inspector</h3>
              </div>
              <span className="section-meta">{activeSources.length} sources</span>
            </div>
            {!activeSources.length ? (
              <div className="panel-empty-state">Source excerpts will appear here after a completed search.</div>
            ) : (
              <div className="source-list">
                {activeSources.map(source => {
                  const isSelected = source.id === selectedSourceId;
                  const normalizedScore = getNormalizedScore(source.rerank_score);
                  return (
                    <article
                      key={source.id}
                      id={`source-${source.id}`}
                      className={`source-card is-${source.section}${isSelected ? ' is-active' : ''}`}
                      onClick={() => setSelectedSourceId(source.id)}
                      style={{ cursor: 'pointer' }}
                    >
                      <div className="source-card-header-row">
                        <div className="source-card-header-left">
                          <span className="source-idx-badge">{source.id}</span>
                          <span className="source-ticker-text">{source.ticker}</span>
                          <span className="source-company-text" title={source.company_name}>
                            {source.company_name}
                          </span>
                        </div>
                        <span className={`source-section-badge is-${source.section}`}>
                          {formatSectionLabel(source.section)}
                        </span>
                      </div>

                      <div className="source-score-row">
                        <div className="source-score-bar-bg">
                          <div
                            className="source-score-bar-fill"
                            style={{ width: `${normalizedScore * 100}%` }}
                          />
                        </div>
                        <span className="source-score-number">{normalizedScore}</span>
                      </div>

                      <span className="source-date-text">{formatFilingDate(source.filing_date)}</span>

                      <p className={`source-text ${isSelected ? 'is-expanded' : 'is-clamped'}`}>
                        {source.text}
                      </p>

                      <div className="source-footer">
                        <span>{source.chunk_id}</span>
                        <span>chars {source.char_start}–{source.char_end}</span>
                      </div>
                    </article>
                  )
                })}
              </div>
            )}
          </aside>
          </Panel>
        </PanelGroup>

        <section className={`history-pane-mobile${mobileTab === 'history' ? ' is-mobile-active' : ''}`}>
          <div className="panel-header">
            <div>
              <p className="section-label">Session</p>
              <h3>Research history</h3>
            </div>
          </div>
          <div className="mobile-history-list">
            {history.length === 0 ? (
              <div className="panel-empty-state">No saved research in this session yet.</div>
            ) : (
              history.map(entry => {
                const isActive = entry.id === activeEntryId;
                return (
                  <div key={entry.id} className="mobile-history-item-container">
                    <button
                      type="button"
                      className={`history-item mobile-history-btn${isActive ? ' is-active' : ''}`}
                      onClick={() => {
                        handleHistorySelect(entry);
                        setMobileTab('answer');
                      }}
                    >
                      <span className="history-item-title">{entry.query}</span>
                      <span className="history-item-meta">
                        {entry.sources?.length || 0} sources · {new Date(entry.createdAt).toLocaleString()}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="delete-history-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteHistoryEntry(entry.id);
                      }}
                      title="Delete report"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M3 6h18" />
                        <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                        <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                      </svg>
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </section>

        <nav className="mobile-tabbar">
          {['answer', 'sources', 'history'].map(tab => (
            <button
              key={tab}
              type="button"
              className={`mobile-tab-button${mobileTab === tab ? ' is-active' : ''}`}
              onClick={() => setMobileTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
      </div>
      </div>
    </div>
  )
}
