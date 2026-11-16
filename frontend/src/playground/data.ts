import type { ProductData, JobRow } from '../data/types';
export interface JobMetadata {id:string;title:string;company:string|null}
export interface PlaygroundData {schema:'skillmap.playground.v1';jobs:JobMetadata[];provenance:{facts_sha256:string;product_sha256:string;source_checks_requested:boolean;salary_grain:string;title_definition:string}}
export function validatePlaygroundData(input:unknown,data:ProductData):PlaygroundData {
  const value=input as PlaygroundData;
  if(!value || value.schema!=='skillmap.playground.v1' || !Array.isArray(value.jobs) || value.jobs.length!==data.rows.length || value.provenance?.facts_sha256!==data.provenance.sourcefacts_sha256 || value.provenance.salary_grain!=='unique_job_id' || value.provenance.source_checks_requested!==true || typeof value.provenance.product_sha256!=='string' || !/^[a-f0-9]{64}$/.test(value.provenance.product_sha256))throw new Error('Title evidence does not match the verified job export');
  const ids=new Set<string>();
  for(const row of value.jobs){if(typeof row.id!=='string'||!row.id||ids.has(row.id)||typeof row.title!=='string'||!row.title.trim()||!(row.company===null||typeof row.company==='string'))throw new Error('Invalid title evidence');ids.add(row.id);}
  return value;
}
export type Section='home'|'skills'|'profiles'|'market';
export interface Selection {section:Section;role:string;title:string;skill:string;source:'adzuna'|'muse';query:string;location:string;marketLocation:string;company:string;marketTitle:string;salaryTitle:string;mapTitle:string;networkMode:'skill'|'title'}
export const defaults:Selection={section:'home',role:'',title:'',skill:'',source:'adzuna',query:'',location:'',marketLocation:'',company:'',marketTitle:'',salaryTitle:'',mapTitle:'',networkMode:'skill'};
export function readSelection():Selection {
  const query=new URLSearchParams(location.search);const slug=location.pathname.split('/')[2]??'home';
  return {...defaults,section:['home','skills','profiles','market'].includes(slug)?slug as Section:'home',role:query.get('role')??'',title:query.get('title')??'',skill:query.get('skill')??'',source:query.get('source')==='muse'?'muse':'adzuna',query:query.get('q')??'',location:query.get('location')??'',company:query.get('company')??'',marketLocation:query.get('marketLocation')??'',marketTitle:query.get('marketTitle')??'',salaryTitle:query.get('salaryTitle')??'',mapTitle:query.get('mapTitle')??'',networkMode:query.get('networkMode')==='title'?'title':'skill'};
}
export function selectionUrl(selection:Selection){const query=new URLSearchParams();for(const [key,value] of Object.entries(selection)){if(key!=='section'&&value!==defaults[key as keyof Selection]&&value!=='')query.set(key==='query'?'q':key,value);}return `/playground${selection.section==='home'?'':`/${selection.section}`}${query.size?`?${query}`:''}`;}
export function quantile(sorted:number[],p:number){if(!sorted.length)return null;const index=(sorted.length-1)*p;const low=Math.floor(index);return sorted[low]+(sorted[Math.ceil(index)]-sorted[low])*(index-low);}
export function titleGroups(data:ProductData,metadata:PlaygroundData|null,role='') {
  const map=new Map<string,{title:string;indices:number[];salaries:number[]}>();
  metadata?.jobs.forEach((job,index)=>{if(role!=='' && data.rows[index][1]!==Number(role))return;const group=map.get(job.title)??{title:job.title,indices:[],salaries:[]};group.indices.push(index);group.salaries.push(data.rows[index][0]);map.set(job.title,group);});
  return [...map.values()].sort((a,b)=>b.indices.length-a.indices.length||a.title.localeCompare(b.title));
}
export function defaultTitle(data:ProductData,metadata:PlaygroundData|null,role='') {const groups=titleGroups(data,metadata,role);return groups.find(group=>group.title==='software engineer'&&group.indices.length>=25)?.title??groups.find(group=>group.indices.length>=25)?.title??groups[0]?.title??'';}
export function cohort(data:ProductData,metadata:PlaygroundData|null,state:Selection):{indices:number[];rows:JobRow[]} {
  const indices=data.rows.flatMap((row,index)=>(state.role!==''&&row[1]!==Number(state.role))||(state.title&&metadata?.jobs[index]?.title!==state.title)||(state.skill&&!row[3].some(index=>data.skills[index].name===state.skill))||(state.company&&metadata?.jobs[index]?.company!==state.company)||(state.location&&(data.locations[row[2]]??'Unrecorded')!==state.location)?[]:[index]);
  return {indices,rows:indices.map(index=>data.rows[index])};
}

// Eligibility protects salary comparisons; it is not a claim of population representativeness.
export const MIN_SALARY_JOBS = 25;
export const FEATURED_TITLE_POOL = ['software engineer','project manager','senior accountant','administrative assistant','data analyst','registered nurse'] as const;
// Related roles must each have >=25 jobs. Skills must occur in >=20% of each title's
// postings. Require >=2 shared prevalent skills and Jaccard >=0.25; no career-path inference.
export const RELATED_TITLE_MIN_JOBS = 25;
export const RELATED_SKILL_PREVALENCE = .2;
export const RELATED_TITLE_MIN_JACCARD = .25;
export const RELATED_TITLE_MIN_SHARED = 2;
export function chooseFeaturedTitle(data:ProductData,metadata:PlaygroundData|null,random:()=>number=Math.random) {
 const eligible=titleGroups(data,metadata).filter(group=>group.indices.length>=MIN_SALARY_JOBS);
 const curated=eligible.filter(group=>FEATURED_TITLE_POOL.includes(group.title as typeof FEATURED_TITLE_POOL[number]));
 const pool=curated.length?curated:eligible.slice(0,8);
 if(!pool.length)return '';
 return pool[Math.min(pool.length-1,Math.max(0,Math.floor(random()*pool.length)))].title;
}
export function skillNeighborhood(data:ProductData,skillIndex:number,limit=11){
 const selected=data.rows.filter(row=>row[3].includes(skillIndex));const counts=new Map<number,number>();
 for(const row of selected)for(const index of row[3])if(index!==skillIndex)counts.set(index,(counts.get(index)??0)+1);
 return [...counts].map(([index,count])=>({index,count,jaccard:count/(selected.length+data.skills[index].count-count)})).sort((a,b)=>b.jaccard-a.jaccard||b.count-a.count||a.index-b.index).slice(0,limit);
}
export function relatedTitleGroups(data:ProductData,groups:ReturnType<typeof titleGroups>,title:string){
 const selected=groups.find(group=>group.title===title);if(!selected||selected.indices.length<RELATED_TITLE_MIN_JOBS)return [];
 const profile=(indices:number[])=>{const counts=new Map<number,number>();for(const index of indices)for(const skill of data.rows[index][3])counts.set(skill,(counts.get(skill)??0)+1);return new Set([...counts].filter(([,count])=>count/indices.length>=RELATED_SKILL_PREVALENCE).map(([index])=>index));};
 const source=profile(selected.indices);if(!source.size)return [];
 return groups.filter(group=>group.title!==title&&group.indices.length>=RELATED_TITLE_MIN_JOBS).map(group=>{const target=profile(group.indices);const shared=[...source].filter(index=>target.has(index)).length;return {...group,shared,jaccard:shared/(source.size+target.size-shared)};}).filter(group=>group.shared>=RELATED_TITLE_MIN_SHARED&&group.jaccard>=RELATED_TITLE_MIN_JACCARD).sort((a,b)=>b.jaccard-a.jaccard||b.indices.length-a.indices.length||a.title.localeCompare(b.title));
}

// Module lifetime is the page visit: SPA section changes never rotate the featured title.
let visitFeatured:{facts:string;title:string}|null=null;
export function featuredTitleForVisit(data:ProductData,metadata:PlaygroundData|null){const facts=data.provenance.sourcefacts_sha256??'';if(!visitFeatured||visitFeatured.facts!==facts)visitFeatured={facts,title:chooseFeaturedTitle(data,metadata)};return visitFeatured.title;}
