export function formatSection(section) {
  if (!section) return 'Document';
  const labels = {
    business: 'Business',
    risk_factors: 'Risk Factors',
    management_discussion: 'Management Discussion',
  }
  return labels[section] || section.replace(/_/g, ' ')
}

export function formatSectionLabel(section) {
  if (!section) return 'DOC';
  const labels = {
    business: 'Business',
    risk_factors: 'Risk',
    management_discussion: 'MD&A',
  }
  return labels[section] || section
}

export function formatDate(dateString) {
  if (!dateString) return 'Unknown filing date'
  const date = new Date(dateString)
  if (isNaN(date.getTime())) return dateString
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).format(date)
}

export function formatFilingDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  if (isNaN(d.getTime())) return dateStr
  return new Intl.DateTimeFormat('en-US', { month: 'short', year: 'numeric' }).format(d)
}

export function formatLongDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  if (isNaN(d.getTime())) return dateStr
  return new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'long', day: 'numeric' }).format(d)
}
