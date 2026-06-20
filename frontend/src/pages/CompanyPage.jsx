import { useEffect, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { formatLongDate } from '../utils'
import './CompanyPage.css'

const SECTIONS = [
  { key: 'business', label: 'Business', item: 'Item 1' },
  { key: 'risk_factors', label: 'Risk Factors', item: 'Item 1A' },
  { key: 'management_discussion', label: 'MD&A', item: 'Item 7' },
]

function wordCount(text) {
  if (!text) return 0
  return text.trim().split(/\s+/).length.toLocaleString()
}

function SectionText({ text, searchTerm }) {
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    setExpanded(false)
  }, [text, searchTerm])

  if (!text) return <p className="cp-empty">This section was not available in the filing.</p>

  const paragraphs = text.split(/\n{2,}/).filter(Boolean)

  let filtered = paragraphs
  if (searchTerm.trim()) {
    const term = searchTerm.toLowerCase()
    filtered = paragraphs.filter(para => para.toLowerCase().includes(term))
  }

  const PREVIEW_COUNT = 8
  const shown = expanded || searchTerm.trim() ? filtered : filtered.slice(0, PREVIEW_COUNT)
  const hasMore = filtered.length > PREVIEW_COUNT && !searchTerm.trim()

  if (filtered.length === 0) {
    return <p className="cp-empty">No paragraphs match your search term.</p>
  }

  return (
    <div className="cp-section-text">
      {shown.map((para, i) => {
        if (!searchTerm.trim()) {
          return <p key={i} className="cp-para">{para.trim()}</p>
        }
        
        const term = searchTerm.trim().replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')
        const regex = new RegExp(`(\\b${term}\\b)`, 'gi')
        const parts = para.split(regex)
        return (
          <p key={i} className="cp-para">
            {parts.map((part, index) => 
              regex.test(part) ? <mark key={index} className="cp-highlight">{part}</mark> : part
            )}
          </p>
        )
      })}
      {hasMore && (
        <button type="button" className="cp-expand-btn" onClick={() => setExpanded(v => !v)}>
          {expanded ? '↑ Show less' : `↓ Show all ${filtered.length} paragraphs (${wordCount(text)} words)`}
        </button>
      )}
    </div>
  )
}

export default function CompanyPage() {
  const { ticker } = useParams()
  const navigate = useNavigate()
  const [filing, setFiling] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('business')
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    setLoading(true)
    setError('')
    setFiling(null)
    fetch(`/api/company/${ticker}`)
      .then(r => {
        if (!r.ok) throw new Error(`No filing found for ${ticker}`)
        return r.json()
      })
      .then(data => setFiling(data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [ticker])

  return (
    <div className="cp-shell">
      <header className="cp-topbar">
        <div className="cp-topbar-left">
          <button type="button" className="cp-back-btn" onClick={() => navigate(-1)}>
            ← Back to terminal
          </button>
          <div className="cp-topbar-sep" />
          {filing && (
            <>
              <span className="cp-topbar-ticker">{filing.ticker}</span>
              <span className="cp-topbar-name">{filing.company_name}</span>
            </>
          )}
        </div>
        <span className="cp-eyebrow">10-K Filing Document</span>
      </header>

      {loading && (
        <div className="cp-loading">
          <div className="cp-loading-spinner" />
          <p>Loading filing for {ticker}…</p>
        </div>
      )}

      {error && (
        <div className="cp-error">
          <p className="cp-error-title">Filing not found</p>
          <p>{error}</p>
          <Link to="/" className="cp-back-link">← Return to terminal</Link>
        </div>
      )}

      {filing && !loading && (
        <div className="cp-body">
          <section className="cp-meta-section">
            <div className="cp-meta-grid">
              <div className="cp-meta-card">
                <span className="cp-meta-label">Ticker</span>
                <span className="cp-meta-val">{filing.ticker}</span>
              </div>
              <div className="cp-meta-card">
                <span className="cp-meta-label">Company</span>
                <span className="cp-meta-val">{filing.company_name}</span>
              </div>
              <div className="cp-meta-card">
                <span className="cp-meta-label">Filing date</span>
                <span className="cp-meta-val">{formatLongDate(filing.filing_date)}</span>
              </div>
              <div className="cp-meta-card">
                <span className="cp-meta-label">CIK</span>
                <span className="cp-meta-val">{filing.cik}</span>
              </div>
              <div className="cp-meta-card">
                <span className="cp-meta-label">Accession</span>
                <span className="cp-meta-val cp-mono">{filing.accession_number || '—'}</span>
              </div>
              <div className="cp-meta-card">
                <span className="cp-meta-label">Sections indexed</span>
                <span className="cp-meta-val">3 of 3</span>
              </div>
            </div>
            <div className="cp-action-row">
              <button
                type="button"
                className="cp-research-btn"
                onClick={() => navigate('/')}
              >
                Research this company →
              </button>
              <a
                href={`https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${filing.cik}&type=10-K`}
                target="_blank"
                rel="noreferrer"
                className="cp-sec-link"
              >
                View on SEC EDGAR ↗
              </a>
            </div>
          </section>

          <div className="cp-tabs">
            {SECTIONS.map(s => {
              const isActive = activeTab === s.key
              return (
                <button
                  key={s.key}
                  type="button"
                  className={`cp-tab cp-tab-${s.key}${isActive ? ' is-active' : ''}`}
                  onClick={() => {
                    setActiveTab(s.key)
                    setSearchTerm('')
                  }}
                >
                  <span className="cp-tab-item">{s.item}</span>
                  <span className="cp-tab-label">{s.label}</span>
                  <span className="cp-tab-words">{wordCount(filing[s.key])} words</span>
                </button>
              )
            })}
          </div>

          <div className="cp-document-view">
            <section className="cp-content">
              <div className="cp-content-header-row">
                <div className="cp-content-header">
                  <div>
                    <p className="cp-eyebrow">{SECTIONS.find(s => s.key === activeTab)?.item}</p>
                    <h2 className="cp-section-title">{SECTIONS.find(s => s.key === activeTab)?.label}</h2>
                  </div>
                </div>
                <div className="cp-section-search">
                  <input
                    type="text"
                    className="cp-search-box"
                    placeholder="Search terms in section..."
                    value={searchTerm}
                    onChange={e => setSearchTerm(e.target.value)}
                  />
                  {searchTerm && (
                    <button type="button" className="cp-clear-search" onClick={() => setSearchTerm('')}>
                      ×
                    </button>
                  )}
                </div>
              </div>

              <div className="cp-quick-tags">
                <span className="cp-tags-label">Quick highlight:</span>
                {['supply chain', 'AI', 'growth', 'competitor', 'risk', 'market'].map(tag => (
                  <button
                    key={tag}
                    type="button"
                    className={`cp-tag-btn${searchTerm.toLowerCase() === tag ? ' is-active' : ''}`}
                    onClick={() => setSearchTerm(searchTerm.toLowerCase() === tag ? '' : tag)}
                  >
                    {tag}
                  </button>
                ))}
              </div>

              <SectionText text={filing[activeTab]} searchTerm={searchTerm} />
            </section>
          </div>
        </div>
      )}
    </div>
  )
}
