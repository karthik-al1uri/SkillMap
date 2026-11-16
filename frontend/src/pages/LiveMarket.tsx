import { useEffect, useMemo, useRef, useState } from 'react';
import type { EChartsOption } from 'echarts';
import type { ProductData } from '../data/types';
import type { MarketData, MarketPosting } from '../data/market';
import { snapshotFreshness } from '../data/market';
import { median, salaryHistogram } from '../data/logic';
import Chart, { chartBase, colors } from '../components/Chart';
import { Badge, EmptyState, Info, PageHeading, Panel, SearchSelect, Segmented, Select, formatMoney, formatNumber, formatPercent } from '../components/UI';

type Source = 'adzuna' | 'muse';
type SkillKind = 'all' | 'technical' | 'general' | 'unknown';
const latestDate = (rows: MarketPosting[]) => [...new Set(rows.map(row => row.date))].sort().at(-1) ?? '';
const countSkills = (rows: MarketPosting[]) => {
  const counts = new Map<string, number>();
  for (const row of rows) for (const skill of new Set(row.skills)) counts.set(skill, (counts.get(skill) ?? 0) + 1);
  return counts;
};
const salaryText = (row: MarketPosting) => row.salary === null ? 'Salary unavailable' : `${formatMoney(row.salary)} / year · ${row.salaryStatus === 'estimated' ? 'estimate' : 'advertised'}`;

export default function LiveMarket({ data, market, marketError }: { data: ProductData; market: MarketData | null; marketError?: string }) {
  const [source, setSource] = useState<Source>('adzuna');
  const [date, setDate] = useState('');
  const [kind, setKind] = useState<SkillKind>('all');
  const [skill, setSkill] = useState('');
  const [salaryKind, setSalaryKind] = useState('all');
  const [salaryLocation, setSalaryLocation] = useState('');
  const [salaryRange, setSalaryRange] = useState('central');
  const [query, setQuery] = useState('');
  const [postingLocation, setPostingLocation] = useState('');
  const [page, setPage] = useState(0);
  const [selectedJob, setSelectedJob] = useState<MarketPosting | null>(null);
  const drawer = useRef<HTMLElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  const currentSource = market?.sources[source];
  const dates = [...new Set(currentSource?.postings.map(row => row.date) ?? [])].sort().reverse();
  const collectionDate = dates.includes(date) ? date : dates[0] ?? '';
  const population = useMemo(() => currentSource?.postings.filter(row => row.date === collectionDate) ?? [], [currentSource, collectionDate]);
  const counts = useMemo(() => countSkills(population), [population]);
  const vocabulary = useMemo(() => new Map(data.skills.map(item => [item.name, item])), [data.skills]);
  const eligibleSkill = (name: string) => kind === 'all' || (kind === 'technical' ? vocabulary.get(name)?.is_technical === true : kind === 'general' ? vocabulary.get(name)?.is_technical === false : vocabulary.get(name)?.is_technical == null);
  const ranking = [...counts].filter(([name]) => eligibleSkill(name)).sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0]));
  const rankingHead = ranking.slice(0,12);
  const outsideSelection = skill ? ranking.find(([name])=>name===skill) : undefined;
  const ranked = (outsideSelection && !rankingHead.some(([name])=>name===skill) ? [...rankingHead.slice(0,11),outsideSelection] : rankingHead).reverse();
  const focused = skill && counts.has(skill) ? skill : '';
  const focusPopulation = useMemo(() => focused ? population.filter(row => row.skills.includes(focused)) : population, [population, focused]);
  const relationships = useMemo(() => {
    if (!focused) return [];
    return [...countSkills(focusPopulation)].filter(([name]) => name !== focused && eligibleSkill(name)).map(([name,count]) => ({name,count,jaccard:count / ((counts.get(name) ?? 0) + focusPopulation.length - count)})).sort((a,b) => b.count-a.count || b.jaccard-a.jaccard).slice(0,6);
  }, [focused, focusPopulation, counts, kind, vocabulary]);
  const freshness = snapshotFreshness(collectionDate);
  const salaryAvailable = population.filter(row => row.salary !== null);
  const advertised = salaryAvailable.filter(row => row.salaryStatus === 'advertised').length;
  const estimated = salaryAvailable.filter(row => row.salaryStatus === 'estimated').length;
  const salaryLocations = [...new Set(focusPopulation.filter(row => row.salary !== null).map(row => row.location ?? 'Unrecorded location'))].sort();
  const salaries = useMemo(() => focusPopulation.filter(row => row.salary !== null && (salaryKind === 'all' || row.salaryStatus === salaryKind) && (!salaryLocation || (row.location ?? 'Unrecorded location') === salaryLocation)), [focusPopulation,salaryKind,salaryLocation]);
  const sortedSalaries = salaries.map(row => row.salary!).sort((a,b)=>a-b);
  const lower = sortedSalaries[Math.floor((sortedSalaries.length-1)*.05)];
  const upper = sortedSalaries[Math.ceil((sortedSalaries.length-1)*.95)];
  const histogram = salaryHistogram(salaries.map(row => [row.salary!,0,0,[],data.tiers.indexOf(row.tier ?? 'Low')]), 20, salaryRange === 'central' && lower < upper ? [lower,upper] : undefined);
  const resultLocations = [...new Set(focusPopulation.map(row => row.location ?? 'Unrecorded location'))].sort();
  const results = useMemo(() => focusPopulation.filter(row => (!postingLocation || (row.location ?? 'Unrecorded location') === postingLocation) && `${row.title} ${row.company ?? ''} ${row.location ?? ''} ${row.skills.join(' ')}`.toLocaleLowerCase().includes(query.toLocaleLowerCase())), [focusPopulation,postingLocation,query]);
  const pageCount = Math.ceil(results.length/6);
  const currentPage = Math.min(page, Math.max(0,pageCount-1));
  const jobs = results.slice(currentPage*6,currentPage*6+6);
  const comparisonSkills = [...(focused ? [[focused,counts.get(focused) ?? 0] as [string,number]] : []), ...ranking.filter(([name])=>name !== focused)].slice(0,8).reverse();
  const comparisonPopulations = (['adzuna','muse'] as const).map(key => {
    const rows = market?.sources[key].postings ?? [];
    const snapshot = key === source ? collectionDate : latestDate(rows);
    const selected = rows.filter(row => row.date === snapshot);
    return {key, label:market?.sources[key].label ?? key, date:snapshot, rows:selected, counts:countSkills(selected)};
  });
  const popularTitles = [...population.reduce((map,row)=>map.set(row.title,(map.get(row.title) ?? 0)+1),new Map<string,number>())].sort((a,b)=>b[1]-a[1]).slice(0,3);
  const resetFocus = () => { setSkill(''); setPostingLocation(''); setSalaryLocation(''); setQuery(''); setPage(0); };
  const focusSkill = (name: string) => { if (name && !eligibleSkill(name)) setKind('all'); setSkill(name); setSalaryLocation(''); setPostingLocation(''); setPage(0); };
  const switchSource = (value: string) => { setSource(value as Source); setDate(''); setKind('all'); setSalaryKind('all'); resetFocus(); setSelectedJob(null); };
  useEffect(() => {
    if (!selectedJob) return;
    closeButton.current?.focus();
    return () => opener.current?.focus();
  }, [selectedJob]);
  const inspectJob = (job: MarketPosting, button: HTMLElement) => { opener.current=button; setSelectedJob(job); };

  const rankingOption: EChartsOption = chartBase({
    grid:{left:8,right:64,top:15,bottom:20,containLabel:true},
    xAxis:{type:'value',minInterval:1,axisLabel:{color:colors.muted},splitLine:{lineStyle:{color:colors.grid,type:'dashed'}}},
    yAxis:{type:'category',data:ranked.map(([name])=>name),axisLabel:{color:colors.text,fontSize:13,width:175,overflow:'break',lineHeight:17},axisTick:{show:false},axisLine:{show:false}},
    tooltip:{trigger:'item',renderMode:'richText',formatter:params=>{const event=params as unknown as {data:{skill:string;value:number}};return `${event.data.skill}\n${formatNumber(event.data.value)} of ${formatNumber(population.length)} postings · ${formatPercent(event.data.value/population.length)}`;}},
    series:[{type:'bar',barMaxWidth:15,data:ranked.map(([name,count])=>({value:count,skill:name,itemStyle:{color:focused===name?colors.palette[1]:colors.accent,opacity:focused && focused!==name ? .65:1}})),label:{show:true,position:'right',color:colors.text,formatter:event=>formatNumber(Number(event.value))},emphasis:{itemStyle:{color:colors.palette[1]}}}],
  });
  const comparisonOption: EChartsOption = chartBase({
    legend:{top:0,left:0,textStyle:{color:colors.muted,fontSize:12},itemWidth:12,itemHeight:10},grid:{left:8,right:25,top:65,bottom:18,containLabel:true},
    xAxis:{type:'value',min:0,max:1,axisLabel:{formatter:value=>formatPercent(Number(value))},splitLine:{lineStyle:{color:colors.grid,type:'dashed'}}},
    yAxis:{type:'category',data:comparisonSkills.map(([name])=>name),axisLabel:{color:colors.text,width:150,overflow:'break',lineHeight:17},axisTick:{show:false},axisLine:{show:false}},
    tooltip:{trigger:'item',renderMode:'richText',formatter:params=>{const event=params as unknown as {data:{name:string;count:number;population:number;value:number[]};seriesName:string};return `${event.data.name}\n${event.seriesName}\n${formatNumber(event.data.count)} / ${formatNumber(event.data.population)} postings · ${formatPercent(event.data.value[0])}`;}},
    series:[{type:'scatter',name:`Historical · ${formatNumber(data.rows.length)} jobs`,symbol:'rect',symbolSize:10,itemStyle:{color:colors.reference},data:comparisonSkills.flatMap(([name],index)=>data.rows.length && vocabulary.has(name) ? [{name,count:vocabulary.get(name)!.count,population:data.rows.length,value:[vocabulary.get(name)!.count/data.rows.length,index]}] : [])},...comparisonPopulations.map((item,index)=>({type:'scatter' as const,name:`${item.label} · ${formatNumber(item.rows.length)} postings`,symbol:index===0?'diamond':'circle',symbolSize:13,itemStyle:{color:colors.palette[index]},data:comparisonSkills.flatMap(([name],rowIndex)=>item.rows.length ? [{name,count:item.counts.get(name) ?? 0,population:item.rows.length,value:[(item.counts.get(name) ?? 0)/item.rows.length,rowIndex]}] : [])}))],
  });
  const salaryOption: EChartsOption = chartBase({
    grid:{left:8,right:20,top:15,bottom:20,containLabel:true},
    xAxis:{type:'category',data:histogram.counts.map((_,index)=>`${formatMoney(histogram.edges[index])} – ${formatMoney(histogram.edges[index+1])}`),axisLabel:{hideOverlap:true,interval:4,formatter:(_value,index)=>formatMoney(histogram.edges[index])}},
    yAxis:{type:'value',minInterval:1,name:'Postings',nameTextStyle:{color:colors.muted},splitLine:{lineStyle:{color:colors.grid,type:'dashed'}}},
    tooltip:{trigger:'axis',renderMode:'richText',valueFormatter:value=>`${formatNumber(Number(value))} unique postings`},series:[{type:'bar',data:histogram.counts,barCategoryGap:'9%',itemStyle:{color:colors.palette[1],borderRadius:[3,3,0,0]}}],
  });

  if (!market) return <div className="page market-page"><PageHeading eyebrow="LIVE MARKET" title="The latest available, in focus." description="Explore saved job-market snapshots."/><EmptyState title="Market snapshot unavailable" description={marketError || 'The verified posting-level export is not available. No postings or salary values are substituted.'}/></div>;

  return <div className="page market-page">
    <PageHeading eyebrow="LIVE MARKET · SAVED INTELLIGENCE" title="The latest available, in focus." description="Explore the skills, roles, and pay in our saved market snapshots."><Badge tone={freshness.stale?'warning':'accent'}>{freshness.stale?'Older cached snapshot':'Cached market snapshot'}</Badge></PageHeading>
    <section className="market-hero" aria-label="Market snapshot summary">
      <div className="market-source-tabs"><Segmented label="Market source" value={source} onChange={switchSource} options={[{value:'adzuna',label:'Adzuna'},{value:'muse',label:'The Muse'}]}/><Info label="About snapshot collection">Dates are collection dates, not publication dates or evidence that a role remains open. No API refresh is enabled. An authorized data owner must refresh source artifacts and regenerate the export.</Info></div>
      <div className="market-snapshot-meta"><div><span>Saved postings</span><strong>{formatNumber(population.length)}</strong><small>{currentSource?.label} · unique posting IDs</small></div><div><span>Collected</span><strong>{collectionDate || 'Unavailable'}</strong><small>{freshness.label}</small></div><div><span>Salary coverage</span><strong>{formatPercent(population.length ? salaryAvailable.length/population.length:null)}</strong><small>{formatNumber(advertised)} advertised · {formatNumber(estimated)} estimated</small></div><div><span>Extracted capabilities</span><strong>{formatNumber(counts.size)}</strong><small>Distinct saved skill mentions</small></div></div>
      <div className={`market-freshness ${freshness.stale?'is-stale':''}`} role="status">{freshness.ageDays === null ? 'Collection freshness cannot be verified. ' : freshness.stale ? 'Older than the 30-day freshness threshold. ' : ''}This is a saved snapshot, not real-time availability. Postings may no longer be open.</div>
      {dates.length>1 && <Select label="Collection date" value={collectionDate} onChange={value=>{setDate(value);resetFocus();}}>{dates.map(value=><option key={value} value={value}>{value}</option>)}</Select>}
    </section>
    {!population.length ? <EmptyState title="No postings in this snapshot" description="No saved records are available for the selected source and date."/> : <>
      <div className="market-explorer">
        <Panel title="What capabilities appear most?" description={`${formatNumber(population.length)} saved ${currentSource?.label} postings · top 12, with selected skill retained`} action={<Info label="About extracted skill mentions">Counts represent postings mentioning a skill, counted once per posting. A mention does not establish proficiency or market-wide demand. Classification uses the saved vocabulary; unclassified skills remain separate.</Info>}>
          <div className="toolbar"><SearchSelect label="Explore a market skill" value={focused} onChange={focusSkill} options={[{value:'',label:'All skills'},...ranking.map(([name,count])=>({value:name,label:`${name} · ${formatNumber(count)}`}))]}/><Select label="Skill classification" value={kind} onChange={value=>{setKind(value as SkillKind);focusSkill('');}}><option value="all">All capabilities</option><option value="technical">Technical skills</option><option value="general">General competencies</option><option value="unknown">Unclassified</option></Select>{focused && <button className="text-button" onClick={resetFocus}>Clear skill focus</button>}</div>
          {ranked.length ? <Chart option={rankingOption} ariaLabel="Cached posting skill ranking" height={Math.max(310,ranked.length*35+40)} onClick={event=>{const item=event.data as {skill?:string}|undefined;if(item?.skill)focusSkill(focused===item.skill?'':item.skill);}}/> : <EmptyState title="No skills in this classification" description="Choose another classification to inspect available mentions."/>}
        </Panel>
        <aside className="market-focus" aria-live="polite">
          {focused ? <><span className="eyebrow">SELECTED CAPABILITY</span><h2>{focused}</h2><div className="market-focus-count"><strong>{formatNumber(focusPopulation.length)}</strong><span>postings · {formatPercent(focusPopulation.length/population.length)} of this snapshot</span></div><h3>Frequently mentioned alongside</h3><div className="market-related">{relationships.length ? relationships.map(item=><button type="button" key={item.name} onClick={()=>focusSkill(item.name)}><span>{item.name}</span><small>{formatNumber(item.count)} shared postings · {formatPercent(item.jaccard)} Jaccard overlap →</small></button>) : <p>No other skills are mentioned in these saved postings.</p>}</div><Info label="About shared mentions">Shared-posting counts and Jaccard overlap are different measures. Jaccard is shared postings divided by postings containing either skill. These descriptive relationships are not validated communities or salary premiums.</Info></> : <><span className="eyebrow">EXPLORE A CONNECTION</span><h2>Start with a skill.</h2><p>Select a bar or search above to see related capabilities and matching postings.</p><h3>Titles in this snapshot</h3><div className="market-related">{popularTitles.map(([title,count])=><button type="button" key={title} onClick={()=>{setQuery(title);setPage(0);document.getElementById('market-results')?.scrollIntoView({behavior:window.matchMedia?.('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});}}><span>{title}</span><small>{formatNumber(count)} postings · inspect saved roles →</small></button>)}</div></>}
        </aside>
      </div>
      <Panel className="market-comparison" title="Same vocabulary. Different populations." description="Historical and cached posting shares · no time trend inferred" action={<Info label="About population comparisons">Each share uses all postings in its own saved population, including postings without matched skills. Historical jobs and the two source snapshots differ in occupation, geography, employer mix, and description coverage. Shared extraction vocabulary does not make the populations directly comparable. Source dates: {comparisonPopulations.map(item=>`${item.label} ${item.date}`).join('; ')}. No growth or decline is inferred.</Info>}>
        {comparisonSkills.length ? <Chart option={comparisonOption} ariaLabel="Historical and source snapshot skill shares" height={Math.max(300,comparisonSkills.length*37+90)}/> : <EmptyState title="No comparable skill mentions"/>}
      </Panel>
      <div className="market-salary-layout"><Panel title="What pay information is available?" description={`${currentSource?.label} · annual USD where supplied`} action={<Info label="About salary reliability">Estimated amounts are source-provided estimates, not verified advertised offers. Muse salary is unavailable in the saved artifacts. Salary summaries use one row per unique posting and do not establish causal skill premiums. Frozen historical tier boundaries remain unchanged.</Info>}>
        {source==='muse' ? <EmptyState title="Salary was not supplied by The Muse" description="Explore Adzuna for the saved estimated and advertised salary observations."/> : <>
          <div className="toolbar"><Select label="Market salary evidence" value={salaryKind} onChange={setSalaryKind}><option value="all">All salary observations</option><option value="advertised">Advertised only</option><option value="estimated">Estimated only</option></Select><SearchSelect label="Salary location" value={salaryLocation} onChange={setSalaryLocation} options={[{value:'',label:'All salary locations'},...salaryLocations.map(value=>({value,label:value}))]}/><Segmented label="Market salary range" value={salaryRange} onChange={setSalaryRange} options={[{value:'central',label:'Central 90%'},{value:'full',label:'Full range'}]}/></div>
          <div className="market-salary-summary" aria-live="polite"><strong>{formatMoney(median(sortedSalaries))}<small>selected median</small></strong><span>{formatNumber(salaries.length)} unique postings{focused?` mentioning ${focused}`:''}</span></div>
          {salaries.length ? <Chart option={salaryOption} ariaLabel="Cached posting salary distribution" height={265}/> : <EmptyState title="No salary observations match" description="Clear the skill, salary-evidence, or location filter."/>}
          {histogram.underflow+histogram.overflow>0 && <p className="market-tail-note">{formatNumber(histogram.underflow)} below and {formatNumber(histogram.overflow)} above the displayed percentile range. Median retains every selected observation.</p>}
        </>}
      </Panel><aside className="market-coverage"><span className="eyebrow">SALARY EVIDENCE</span><h2>{source==='muse'?'No inferred pay.':'Estimates stay visible.'}</h2><dl><div><dt>Advertised</dt><dd>{formatNumber(advertised)}</dd></div><div><dt>Estimated</dt><dd>{formatNumber(estimated)}</dd></div><div><dt>Not supplied</dt><dd>{formatNumber(population.length-salaryAvailable.length)}</dd></div></dl>{source==='adzuna' && <><h3>Observed salary-tier mix</h3><div className="market-tier-strip" aria-label="Selected salary tier composition">{data.tiers.map((tier,index)=>{const n=salaries.filter(row=>row.tier===tier).length;return <div key={tier}><span style={{borderLeft:`3px solid ${colors.tiers[index]}`}}>{tier}</span><strong>{formatNumber(n)} <small>{formatPercent(salaries.length?n/salaries.length:null)}</small></strong></div>;})}</div></>}<small>Coverage uses the full selected source/date. Tier counts use the salary filters.</small></aside></div>
      <section className="market-results" id="market-results" aria-labelledby="market-results-heading"><div className="market-results-heading"><div><span className="eyebrow">BEHIND THE SNAPSHOT</span><h2 id="market-results-heading">Explore the saved postings</h2></div><Badge>{formatNumber(results.length)} matching records</Badge></div><div className="toolbar"><label className="market-posting-search">Search postings<input type="search" aria-label="Search saved postings" placeholder="Title, company, location, or skill" value={query} onChange={event=>{setQuery(event.target.value);setPage(0);}}/></label><SearchSelect label="Posting location" value={postingLocation} onChange={value=>{setPostingLocation(value);setPage(0);}} options={[{value:'',label:'All posting locations'},...resultLocations.map(value=>({value,label:value}))]}/>{(focused||query||postingLocation) && <button type="button" className="text-button" onClick={resetFocus}>Reset posting filters</button>}</div>
        {jobs.length ? <div className="market-posting-grid">{jobs.map(job=><article className="market-posting" key={job.id}><span className="market-posting-source">{currentSource?.label} · {job.date}</span><h3>{job.title}</h3><p>{job.company ?? 'Company not recorded'}<br/>{job.location ?? 'Location not recorded'}</p><strong>{salaryText(job)}</strong><div className="market-skill-tags">{job.skills.slice(0,4).map(name=><button type="button" key={name} onClick={()=>focusSkill(name)} aria-label={`Focus skill ${name}`}>{name}</button>)}{job.skills.length>4 && <span>+{job.skills.length-4}</span>}</div><button type="button" className="text-button" onClick={event=>inspectJob(job,event.currentTarget)}>Inspect saved posting →</button></article>)}</div> : <EmptyState title="No saved postings match" description="Try a broader search or reset the posting filters."/>}
        {pageCount>1 && <div className="market-pagination"><button type="button" className="ghost-button" disabled={currentPage===0} onClick={()=>setPage(currentPage-1)}>Previous</button><span aria-live="polite">Page {currentPage+1} of {pageCount}</span><button type="button" className="ghost-button" disabled={currentPage===pageCount-1} onClick={()=>setPage(currentPage+1)}>Next</button></div>}
      </section>
      <details className="detail-disclosure"><summary>Accessible chart values & snapshot provenance</summary><table><caption>Skill mentions · selected source/date</caption><thead><tr><th>Skill</th><th>Postings</th><th>Share</th><th>Focus</th></tr></thead><tbody>{[...ranked].reverse().map(([name,count])=><tr key={name}><th>{name}</th><td>{formatNumber(count)}</td><td>{formatPercent(count/population.length)}</td><td><button type="button" className="text-button" onClick={()=>focusSkill(name)}>Inspect {name}</button></td></tr>)}</tbody></table><table><caption>Comparison population shares</caption><thead><tr><th>Skill</th><th>Historical</th>{comparisonPopulations.map(item=><th key={item.key}>{item.label}</th>)}</tr></thead><tbody>{comparisonSkills.map(([name])=><tr key={name}><th>{name}</th><td>{formatPercent(data.rows.length && vocabulary.has(name)?vocabulary.get(name)!.count/data.rows.length:null)}</td>{comparisonPopulations.map(item=><td key={item.key}>{formatPercent(item.rows.length?(item.counts.get(name) ?? 0)/item.rows.length:null)}</td>)}</tr>)}</tbody></table><table><caption>Displayed salary distribution · annual USD</caption><thead><tr><th>Range</th><th>Postings</th></tr></thead><tbody>{histogram.counts.map((count,index)=><tr key={index}><th>{formatMoney(histogram.edges[index])} – {formatMoney(histogram.edges[index+1])}</th><td>{formatNumber(count)}</td></tr>)}</tbody></table><p>Read-only export of saved cleaned postings. No refresh or external API request occurs. {currentSource?.limitations.join(' ')}</p><p>{currentSource?.provenance.path}<br/>SHA-256: {currentSource?.provenance.sha256}</p></details>
    </>}
    {selectedJob && <><button className="market-drawer-backdrop" type="button" aria-label="Close posting details" onClick={()=>setSelectedJob(null)}/><aside className="market-drawer" ref={drawer} role="dialog" aria-modal="true" aria-labelledby="market-posting-title" onKeyDown={event=>{
      if(event.key==='Escape'){event.stopPropagation();setSelectedJob(null);}
      if(event.key==='Tab'){
        const nodes=[...drawer.current?.querySelectorAll<HTMLElement>('button, a, input, select, [tabindex="0"]') ?? []];
        const first=nodes[0],last=nodes.at(-1);
        if(event.shiftKey&&document.activeElement===first){event.preventDefault();last?.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first?.focus();}
      }
    }}><button ref={closeButton} type="button" className="ghost-button" onClick={()=>setSelectedJob(null)}>Close details</button><span className="eyebrow">{currentSource?.label} · COLLECTED {selectedJob.date}</span><h2 id="market-posting-title">{selectedJob.title}</h2><p>{selectedJob.company ?? 'Company not recorded'}<br/>{selectedJob.location ?? 'Location not recorded'}</p><strong>{salaryText(selectedJob)}</strong><Badge>{selectedJob.salaryStatus==='unavailable'?'No salary supplied':selectedJob.salaryStatus==='estimated'?'Source estimate · not an advertised offer':'Advertised salary in saved source'}</Badge><h3>Extracted skill mentions</h3><div className="market-skill-tags">{selectedJob.skills.length?selectedJob.skills.map(name=><button type="button" key={name} onClick={()=>{focusSkill(name);setSelectedJob(null);}}>{name}</button>):<p>No matched skills in this saved posting.</p>}</div><p className="subtle">Saved ID {selectedJob.id}. Collection date does not verify that this role remains open. The saved dataset does not provide a verified application link.</p></aside></>}
  </div>;
}
