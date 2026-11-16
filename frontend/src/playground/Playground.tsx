import {useEffect,useRef,useState} from 'react';
import type {ProductData} from '../data/types';
import type {MarketData} from '../data/market';
import {validateData} from '../data/logic';
import {validateMarketData} from '../data/market';
import Icon from '../components/Icon';
import type {IconName} from '../components/Icon';
import {navigate} from '../routes';
import type {PlaygroundData,Selection} from './data';
import {defaults,readSelection,selectionUrl,validatePlaygroundData,featuredTitleForVisit} from './data';
import Home from './Home';
import SkillsSalary from './SkillsSalary';
import Profiles from './Profiles';
import Market from './Market';
import {Empty} from './Visuals';
import './playground.css';
const sections=[{key:'home',label:'Home',icon:'overview'},{key:'skills',label:'Skills & Salary Intelligence',icon:'skills'},{key:'profiles',label:'Career Profiles',icon:'clusters'},{key:'market',label:'Live Market',icon:'live'}] as const;
export default function Playground(){
 const [state,setState]=useState(readSelection),[data,setData]=useState<ProductData|null>(null),[market,setMarket]=useState<MarketData|null>(null),[metadata,setMetadata]=useState<PlaygroundData|null>(null),[error,setError]=useState(''),[attempt,setAttempt]=useState(0),[loaded,setLoaded]=useState(false);const main=useRef<HTMLElement>(null);const featured=useRef<string|null>(null);const [featuredTitle,setFeaturedTitle]=useState('');
 useEffect(()=>{const sync=()=>{setState(readSelection());};window.addEventListener('popstate',sync);return()=>window.removeEventListener('popstate',sync);},[]);
 useEffect(()=>{let active=true;const controller=new AbortController();setError('');setLoaded(false);const read=async(name:string)=>{const response=await fetch(`${import.meta.env.BASE_URL}data/${name}.json`,{signal:controller.signal});if(!response.ok)throw new Error('Saved data unavailable');return response.json();};
  Promise.allSettled([read('skillmap'),read('market'),read('playground')]).then(results=>{if(!active)return;try{if(results[0].status!=='fulfilled')throw new Error('The verified job export is unavailable.');const product=validateData(results[0].value);setData(product);if(results[1].status==='fulfilled'){try{setMarket(validateMarketData(results[1].value));}catch{setMarket(null);}}else setMarket(null);if(results[2].status==='fulfilled'){try{const titles=validatePlaygroundData(results[2].value,product);setMetadata(titles);if(featured.current===null)featured.current=featuredTitleForVisit(product,titles);setFeaturedTitle(featured.current);}catch{setMetadata(null);}}else setMetadata(null);}catch(reason){setError(reason instanceof Error?reason.message:'Saved data unavailable');}finally{setLoaded(true);}});return()=>{active=false;controller.abort();};
 },[attempt]);
 const update=(patch:Partial<Selection>)=>{const next={...state,...patch};if(patch.title!==undefined&&patch.title!==state.title&&patch.company===undefined)next.company='';setState(next);navigate(selectionUrl(next),patch.query!==undefined&&patch.section===undefined);if(patch.section&&patch.section!==state.section){main.current?.focus();window.scrollTo?.({top:0,behavior:'instant'});}};
 return <div className="pg"><a className="pg-skip" href="#pg-main" onClick={event=>{event.preventDefault();main.current?.focus();}}>Skip to content</a><header className="pg-header"><a className="pg-brand" href="/playground" onClick={event=>{event.preventDefault();update({...defaults});}}>Skill<span>Map</span><small>Playground</small></a><nav aria-label="Playground navigation">{sections.map(section=><a key={section.key} href={selectionUrl({...state,section:section.key})} aria-current={state.section===section.key?'page':undefined} onClick={event=>{if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();update({section:section.key});}}><Icon name={section.icon as IconName} size={17}/><span>{section.label}</span></a>)}</nav></header><main id="pg-main" ref={main} tabIndex={-1} className="pg-main">{error?<><Empty>{error}</Empty><button className="pg-button" onClick={()=>setAttempt(value=>value+1)}>Retry</button></>:!loaded||!data?<div className="pg-loading" role="status">Loading saved data…<div/><div/></div>:state.section==='home'?<Home data={data} market={market} state={state} update={update}/>:state.section==='skills'?<SkillsSalary data={data} metadata={metadata} state={state} update={update} featuredTitle={featuredTitle}/>:state.section==='profiles'?<Profiles data={data} metadata={metadata} state={state} update={update}/>:<Market market={market} state={state} update={update}/>}</main></div>;
}
