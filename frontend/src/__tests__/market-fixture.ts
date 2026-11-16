/** Explicitly synthetic interaction fixture; never imported by production code. */
import type { MarketData } from '../data/market';
export function marketFixture(): MarketData {
  const reference = {path:'synthetic-snapshot.csv',bytes:0,sha256:'0'.repeat(64)};
  return {schema:'skillmap.market.v1',sources:{
    adzuna:{label:'Adzuna',provenance:reference,limitations:['Synthetic test fixture'],postings:[
      {id:'a1',title:'Synthetic data engineer',company:'Example Co',location:'Boulder',date:'2026-10-14',skills:['python','sql'],salary:100000,salaryStatus:'advertised',tier:'Mid'},
      {id:'a2',title:'Synthetic analyst',company:'Example Two',location:'Denver',date:'2026-10-14',skills:['python'],salary:70000,salaryStatus:'estimated',tier:'Mid'},
    ]},
    muse:{label:'The Muse',provenance:reference,limitations:['No saved salary fields'],postings:[
      {id:'m1',title:'Synthetic platform engineer',company:'Example Three',location:'Remote',date:'2026-10-14',skills:['python','sql'],salary:null,salaryStatus:'unavailable',tier:null},
    ]},
  },provenance:{approved_html:reference,source_checks_requested:true,execution:'Synthetic fixture only',posting_grain:'(source, id, collection date)',salary_currency:'USD',salary_period:'annual',refresh_status:'No refresh executed'}};
}
