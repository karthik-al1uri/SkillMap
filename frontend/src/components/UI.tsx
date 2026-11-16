import { useId, useState } from 'react';
import type { ReactNode } from 'react';
export const formatMoney = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);
export const formatNumber = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('en-US').format(value);
export const formatScore = (value: number | null | undefined) => value == null ? '—' : value.toFixed(3);
export const formatPercent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'accent' | 'warning' | 'neutral' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}
export function PageHeading({ eyebrow, title, description, children }: { eyebrow: string; title: string; description?: string; children?: ReactNode }) {
  return <header className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1>{description && <p>{description}</p>}</div>{children && <div className="heading-actions">{children}</div>}</header>;
}
export function StatCard({ label, value, detail, accent = false }: { label: string; value: ReactNode; detail?: string; accent?: boolean }) {
  return <article className={`stat-card ${accent ? 'stat-accent' : ''}`}><div className="stat-label">{label}</div><div className="stat-value">{value}</div>{detail && <div className="stat-detail">{detail}</div>}</article>;
}
export function Panel({ title, description, children, className = '', action }: { title: string; description?: string; children: ReactNode; className?: string; action?: ReactNode }) {
  return <section className={`panel-card ${className}`}><header className="panel-heading"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{action}</header><div className="panel-content">{children}</div></section>;
}
export function EmptyState({ title, description }: { title: string; description?: string }) {
  return <div className="empty-state" role="status"><span aria-hidden="true" className="empty-mark">◇</span><h3>{title}</h3>{description && <p>{description}</p>}</div>;
}
export function Select({ label, value, onChange, children }: { label: string; value: string | number; onChange: (value: string) => void; children: ReactNode }) {
  return <label className="select-control"><span>{label}</span><select value={value} onChange={event => onChange(event.target.value)}>{children}</select></label>;
}
export function Segmented({ label, value, options, onChange }: { label: string; value: string; options: { value: string; label: string }[]; onChange: (value: string) => void }) {
  return <div className="segmented" role="group" aria-label={label}>{options.map(option => <button key={option.value} type="button" aria-pressed={value === option.value} onClick={() => onChange(option.value)}>{option.label}</button>)}</div>;
}
export function Insight({ children }: { children: ReactNode }) { return <div className="insight"><span className="insight-mark" aria-hidden="true">i</span><p>{children}</p></div>; }

/** Searchable single-selection control with a keyboard-operable listbox. */
export function SearchSelect({ label, value, onChange, options, placeholder = 'Search…' }: {
  label: string; value: string; onChange: (value: string) => void;
  options: { value: string; label: string }[]; placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const id = useId();
  const selected = options.find(option => option.value === value);
  const matches = options.filter(option => option.label.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const choose = (next: string) => { onChange(next); setOpen(false); setQuery(''); setActive(0); };
  return <div className="search-select" onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false); }}>
    <label htmlFor={id}>{label}</label>
    <div className="search-select-input"><input id={id} role="combobox" aria-autocomplete="list" aria-expanded={open} aria-controls={`${id}-options`} aria-activedescendant={open && matches[active] ? `${id}-${active}` : undefined}
      autoComplete="off" value={open ? query : selected?.label ?? ''} placeholder={selected?.label ?? placeholder}
      onFocus={() => { setOpen(true); setQuery(''); setActive(0); }}
      onClick={() => { if (!open) { setQuery(''); setActive(0); } setOpen(true); }}
      onChange={event => { setQuery(event.target.value); setOpen(true); setActive(0); }}
      onKeyDown={event => {
        if (event.key === 'Escape') { event.stopPropagation(); setOpen(false); setQuery(''); setActive(0); }
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
          event.preventDefault(); setOpen(true);
          setActive(index => matches.length ? (index + (event.key === 'ArrowDown' ? 1 : -1) + matches.length) % matches.length : 0);
        }
        if (event.key === 'Enter' && open) { event.preventDefault(); if (matches[active]) choose(matches[active].value); }
        if (event.key === 'Tab') setOpen(false);
      }}/><span aria-hidden="true">⌕</span></div>
    {open && <div className="search-results" role="listbox" id={`${id}-options`} aria-label={`${label} options`}>
      {matches.length ? matches.map((option,index) => <button type="button" role="option" id={`${id}-${index}`} key={option.value} tabIndex={-1} aria-selected={value === option.value} className={index === active ? 'is-active' : ''} onMouseEnter={() => setActive(index)} onMouseDown={event => event.preventDefault()} onClick={() => choose(option.value)}>{option.label}{value === option.value && <span aria-hidden="true">✓</span>}</button>) : <p role="status">No matching options</p>}
    </div>}
  </div>;
}

/** Keyboard-operable contextual help, with no hover-only information. */
export function Info({ label, children }: { label: string; children: ReactNode }) {
  return <details className="info-tooltip"><summary aria-label={label} title={label}><span aria-hidden="true">i</span></summary><div role="note"><strong>{label}</strong>{children}</div></details>;
}
