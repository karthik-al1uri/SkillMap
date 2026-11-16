import {lazy,Suspense,useEffect,useState} from 'react';
import App from './App';
const Playground=lazy(()=>import('./playground/Playground'));
/** Only the isolated prototype gets this shell. Production App remains unchanged. */
export default function Entry(){
 const [playground,setPlayground]=useState(()=>/^\/playground(?:\/|$)/.test(location.pathname));
 useEffect(()=>{const sync=()=>setPlayground(/^\/playground(?:\/|$)/.test(location.pathname));window.addEventListener('popstate',sync);return()=>window.removeEventListener('popstate',sync);},[]);
 return playground?<Suspense fallback={<div role="status">Loading playground…</div>}><Playground/></Suspense>:<App/>;
}
