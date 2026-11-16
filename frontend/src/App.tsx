import { useEffect, useRef, useState } from 'react';
import Overview from './pages/Overview';
import Skills from './pages/Skills';
import Salary from './pages/Salary';
import Clusters from './pages/Clusters';
import Live from './pages/Live';
import LiveMarket from './pages/LiveMarket';
import Model from './pages/Model';
import { AdminOverview, AssociationMining, DataQuality } from './pages/Admin';
import Icon from './components/Icon';
import { EmptyState, Segmented } from './components/UI';
import { validateData } from './data/logic';
import { validateMarketData } from './data/market';
import type { MarketData } from './data/market';
import type { ProductData } from './data/types';
import { initialRoute, legacyRoutes, navigate, resolveRoute, routes } from './routes';
export default function App() {
  const [route, setRoute] = useState(initialRoute);
  const [data, setData] = useState<ProductData | null>(null);
  const [market, setMarket] = useState<MarketData | null>(null);
  const [marketError, setMarketError] = useState('');
  const [marketLoading, setMarketLoading] = useState(true);
  const [error, setError] = useState('');
  const [attempt, setAttempt] = useState(0);
  const [open, setOpen] = useState(false);
  const heading = useRef<HTMLElement>(null);
  const menu = useRef<HTMLButtonElement>(null);
  const mode = route?.mode ?? (location.pathname.startsWith('/admin') ? 'admin' : 'user');
  const navigation = routes.filter(item=>item.mode===mode);
  useEffect(() => {
    const sync = () => {
      const legacy = legacyRoutes[location.hash.slice(1)];
      if (legacy) window.history.replaceState(null,'',legacy);
      setRoute(resolveRoute(location.pathname)); setOpen(false); heading.current?.focus();
    };
    window.addEventListener('popstate', sync); window.addEventListener('hashchange', sync);
    if (location.pathname === '/') window.history.replaceState(null,'','/app');
    return () => { window.removeEventListener('popstate',sync); window.removeEventListener('hashchange',sync); };
  }, []);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    setError(''); setMarketError(''); setMarketLoading(true);
    fetch(`${import.meta.env.BASE_URL}data/skillmap.json`, {signal:controller.signal})
      .then(response=>{if(!response.ok) throw new Error(`Dataset unavailable (HTTP ${response.status}). Export the approved saved facts bundle.`);return response.json();})
      .then(value=>{const valid=validateData(value);if(active)setData(valid);})
      .catch((reason:unknown)=>{if(active)setError(reason instanceof Error?reason.message:'Unable to load saved evidence.');});
    fetch(`${import.meta.env.BASE_URL}data/market.json`, {signal:controller.signal})
      .then(response=>{if(!response.ok)throw new Error(`Cached posting data unavailable (HTTP ${response.status}).`);return response.json();})
      .then(value=>{const valid=validateMarketData(value);if(active)setMarket(valid);})
      .catch((reason:unknown)=>{if(active){setMarket(null);setMarketError(reason instanceof Error?reason.message:'Cached posting data unavailable.');}})
      .finally(()=>{if(active)setMarketLoading(false);});
    return()=>{active=false;controller.abort();};
  }, [attempt]);
  useEffect(()=>{if(open)document.querySelector<HTMLAnchorElement>('#sidebar nav a')?.focus();},[open]);
  const closeMenu=()=>{setOpen(false);menu.current?.focus();};
  const page = () => {
    if (!data || !route) return null;
    switch(route.page) {
      case 'explore': return <Overview data={data}/>;
      case 'skills': return <Skills data={data}/>;
      case 'salary': return <Salary data={data}/>;
      case 'profiles': return <Clusters data={data}/>;
      case 'market': return marketLoading ? <div role="status" className="market-loading">Loading cached posting snapshots…<div className="skeleton chart-skeleton"/></div> : <LiveMarket data={data} market={market} marketError={marketError}/>;
      case 'admin-overview': return <AdminOverview data={data}/>;
      case 'model': return <Model data={data}/>;
      case 'mining': return <AssociationMining data={data}/>;
      case 'clustering': return <Clusters data={data} technical/>;
      case 'quality': return <DataQuality data={data}/>;
      case 'validation': return <Live data={data}/>;
    }
  };
  return <div className={`app-shell mode-${mode}`} onClick={event=>{
    if(event.defaultPrevented || event.button!==0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey)return;
    const link=(event.target as Element).closest('a');
    if(!link || link.target || link.hasAttribute('download'))return;
    const url=new URL(link.href,location.href);
    if(url.origin===location.origin && /^\/(app|admin)(\/|$)/.test(url.pathname)) {event.preventDefault();navigate(url.pathname);}
  }} onKeyDown={event=>{
    if(event.key==='Escape' && open)closeMenu();
    if(event.key==='Tab' && open){const controls=[menu.current,...document.querySelectorAll<HTMLAnchorElement>('#sidebar nav a'),...document.querySelectorAll<HTMLButtonElement>('#sidebar .mode-switch button')].filter((item):item is HTMLButtonElement|HTMLAnchorElement=>item!==null);const index=controls.indexOf(document.activeElement as HTMLAnchorElement);event.preventDefault();controls[(index+(event.shiftKey?-1:1)+controls.length)%controls.length]?.focus();}
  }}>
    <a className="skip-link" href="#main-content" onClick={event=>{event.preventDefault();heading.current?.focus();}}>Skip to content</a>
    {open && <button className="nav-backdrop" aria-label="Close navigation" onClick={closeMenu}/>}
    <aside id="sidebar" className={`sidebar ${open?'is-open':''}`}>
      <div className="brand"><span className="brand-mark" aria-hidden="true"><i/><i/><i/></span>Skill<span className="brand-suffix">Map</span></div>
      <div className="mode-switch"><Segmented label="Application mode" value={mode} options={[{value:'user',label:'User'},{value:'admin',label:'Admin'}]} onChange={value=>navigate(value==='admin'?'/admin':'/app')}/></div>
      <div className="nav-label">{mode==='user'?'YOUR NEXT DIRECTION':'RESEARCH & VALIDATION'}</div>
      <nav aria-label="Main navigation" className="main-nav" onKeyDown={event=>{if(!['ArrowDown','ArrowUp','Home','End'].includes(event.key))return;event.preventDefault();const links=Array.from(event.currentTarget.querySelectorAll('a'));const index=links.indexOf(document.activeElement as HTMLAnchorElement);links[event.key==='Home'?0:event.key==='End'?links.length-1:(index+(event.key==='ArrowDown'?1:-1)+links.length)%links.length]?.focus();}}>
        {navigation.map(item=><a key={item.path} className={`nav-item ${item.page==='market'?'nav-market':''}`} href={item.path} aria-current={route?.path===item.path?'page':undefined}><Icon name={item.icon} size={18}/><span>{item.label}</span>{item.page==='market' && <small>SNAPSHOT</small>}</a>)}
      </nav>
      <div className="sidebar-bottom"><div className="local-status"><i/>Saved evidence</div><p>{mode==='user'?'Explore skills. Understand your options.':'Presentation mode · no authentication'}</p></div>
    </aside>
    <div className="workspace"><header className="topbar"><button ref={menu} className="icon-button mobile-menu" aria-label={open?'Close menu':'Open menu'} aria-expanded={open} aria-controls="sidebar" onClick={()=>setOpen(!open)}><Icon name="menu"/></button><div className="breadcrumb">{mode==='user'?'Career intelligence':'Research console'}<span>/</span><strong>{route?.label??'Page not found'}</strong></div><span className="source-freshness">{mode==='user'?<a href="/app/live">Latest available market snapshot ↗</a>:<span>Historical · grouped · cached</span>}</span></header>
      <main id="main-content" ref={heading} tabIndex={-1} className="main-content">
        {!route?<div className="error-screen"><EmptyState title="Page not found" description="Choose a destination from the navigation."/><a href="/app" className="button-primary">Explore SkillMap</a></div>:error?<div className="error-screen"><EmptyState title="Saved evidence unavailable" description="No substitute data is used."/><p role="alert">{error}</p><button className="button-primary" onClick={()=>setAttempt(value=>value+1)}>Retry data loading</button></div>:data?page():<div className="loading-layout" role="status" aria-label="Loading saved evidence"><p className="loading-label">Loading saved evidence…</p><div className="skeleton stat-skeleton"/><div className="skeleton chart-skeleton"/></div>}
        {data && <footer className="footer"><span>{mode==='user'?'Observed postings · explore with context':'Research presentation · not an access-control boundary'}</span><details><summary>Data provenance</summary>{mode==='admin'?<><p>Facts SHA-256: {data.provenance.sourcefacts_sha256??'Not supplied'}</p><ul>{Object.entries(data.provenance.sources).map(([id,source])=><li key={id}>{source.status}: {source.path}<br/><code>{source.sha256}</code></li>)}</ul></>:<p>Historical jobs and cached market snapshots are separate populations. No real-time collection or automatic refresh occurs.</p>}{data.provenance.recovery_notice!=null&&<p>{String(data.provenance.recovery_notice)}</p>}</details></footer>}
      </main>
    </div>
  </div>;
}
