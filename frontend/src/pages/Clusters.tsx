import { useMemo, useState } from 'react';
import type { EChartsOption } from 'echarts';
import type { ProductData } from '../data/types';
import { salaryHistogram } from '../data/logic';
import Chart, { chartBase, colors } from '../components/Chart';
import { Badge, EmptyState, PageHeading, Panel, Segmented, Select, Info, StatCard, formatMoney, formatNumber, formatPercent, formatScore } from '../components/UI';

const representationName = (name: string) => name === 'binary' ? 'Binary' : name === 'svd_l2' ? 'SVD32 + L2' : name;

export default function Clusters({ data, technical = false }: { data: ProductData; technical?: boolean }) {
  const [range, setRange] = useState('central');
  const [selection, setSelection] = useState(() => String(Math.max(0, data.roles.findIndex(role => role.name === 'Data, Tech & Engineering'))));
  const roleIndex = data.roles[Number(selection)] ? Number(selection) : 0;
  const role = data.roles[roleIndex];
  const members = useMemo(() => data.rows.filter(row => row[1] === roleIndex), [data.rows, roleIndex]);
  const histogram = useMemo(() => {
    const sorted = members.map(row => row[0]).sort((a,b) => a-b);
    const low = sorted[Math.floor((sorted.length-1)*.05)];
    const high = sorted[Math.ceil((sorted.length-1)*.95)];
    return salaryHistogram(members, 18, range === 'central' && low < high ? [low,high] : undefined);
  }, [members, range]);
  const signature = role ? [...role.top_skills].slice(0, 8).reverse() : [];
  const robustness = data.robustness;
  const summaries = [...(robustness?.summary ?? [])].sort((a,b) => a.k - b.k || a.representation.localeCompare(b.representation));
  const skillOption: EChartsOption = chartBase({
    legend: { top: 0, left: 0, textStyle: { color: colors.muted, fontSize: 12 }, itemWidth: 10, itemHeight: 10 },
    grid: { left: 8, right: 32, top: 52, bottom: 18, containLabel: true },
    xAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: colors.muted, fontSize: 12, formatter: value => formatPercent(Number(value)) }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'category', data: signature.map(skill => skill.skill), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: colors.text, fontSize: 12 } },
    tooltip: { trigger: 'item', renderMode: 'richText', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, formatter: event => {
      const item = Array.isArray(event) ? event[0] : event;
      const values = item.value as number[];
      return `${signature[values[1]]?.skill ?? ''}\n${item.seriesName}: ${formatPercent(values[0])}`;
    } },
    series: [
      { type: 'scatter', name: 'Full saved cohort · diamond', symbol: 'diamond', symbolSize: 11, itemStyle: { color: colors.reference }, data: signature.flatMap((skill,index) => skill.overall_share == null ? [] : [[skill.overall_share,index]]) },
      { type: 'scatter', name: 'Selected archetype · circle', symbol: 'circle', symbolSize: 17, itemStyle: { color: colors.accent, borderColor: '#e7ddff', borderWidth: 1 }, data: signature.map((skill,index) => [skill.share,index]) },
    ],
  });
  const salaryOption: EChartsOption = chartBase({
    grid: { left: 8, right: 15, top: 16, bottom: 18, containLabel: true },
    xAxis: { type: 'category', data: histogram.counts.map((_,index) => `${formatMoney(histogram.edges[index])} – ${formatMoney(histogram.edges[index+1])}`), axisTick: { show: false }, axisLabel: { color: colors.muted, hideOverlap: true, formatter: (_value,index) => formatMoney(histogram.edges[index]) } },
    yAxis: { type: 'value', minInterval: 1, axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    tooltip: { trigger: 'axis', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, valueFormatter: value => `${formatNumber(Number(value))} jobs` },
    series: [{ type: 'bar', data: histogram.counts, barCategoryGap: '12%', itemStyle: { color: colors.accent, opacity: .8, borderRadius: [3,3,0,0] } }],
  });
  const robustnessOption: EChartsOption = chartBase({
    grid: { left: 8, right: 48, top: 12, bottom: 18, containLabel: true },
    xAxis: { type: 'value', min: Math.min(0,...summaries.map(row => row.mean_cosine_silhouette_original)), axisLabel: { color: colors.muted, formatter: value => Number(value).toFixed(2) }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'category', inverse: true, data: summaries.map(row => `${representationName(row.representation)} · k=${row.k}`), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: colors.text, fontSize: 12 } },
    tooltip: { trigger: 'axis', renderMode: 'richText', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, valueFormatter: value => formatScore(Number(value)) },
    series: [{ type: 'bar', name: 'Original-space cosine silhouette', data: summaries.map(row => ({ value: row.mean_cosine_silhouette_original, itemStyle: { color: row.representation === 'binary' ? colors.reference : colors.accent } })), barMaxWidth: 20, label: { show: true, position: 'right', color: colors.text, fontSize: 12, formatter: event => formatScore(Number(event.value)) }, itemStyle: { borderRadius: [0,3,3,0] } }],
  });
  return <div className="page cluster-page">
    <PageHeading eyebrow={technical ? "SAVED JOB ARCHETYPES" : "CAREER PROFILES"} title={technical ? "Different roles. Shared capabilities." : "Find a profile to explore."} description="Explore characteristic skills and observed pay in saved job profiles."><Badge tone="warning">{technical ? "Exploratory · weak separation" : "Exploratory profiles · not validated occupations"}</Badge></PageHeading>
    {!role ? <EmptyState title="No saved archetypes" description="Cluster profiles were not supplied with these artifacts." /> : <>
      <div className="archetype-picker" role="group" aria-label="Archetype profiles">
        {data.roles.map((item,index) => <button type="button" key={item.id} className="archetype-option" aria-pressed={roleIndex === index} onClick={() => setSelection(String(index))}>
          <span>{item.name}</span><strong>{formatNumber(item.count)} <small>jobs · {formatPercent(data.rows.length ? item.count/data.rows.length : null)}</small></strong>
          <span className="archetype-meter" aria-hidden="true"><i style={{ display: 'block', height: 3, width: `${data.rows.length ? item.count/data.rows.length*100 : 0}%`, background: colors.palette[index % colors.palette.length] }}/></span>
        </button>)}
      </div>
      <div className="toolbar"><Select label="Explore an archetype" value={String(roleIndex)} onChange={setSelection}>{data.roles.map((item,index) => <option key={item.id} value={index}>{item.name}</option>)}</Select><p className="subtle">Saved assignments · select a profile to update all panels</p></div>
      <div className="cluster-summary" aria-live="polite">
        <StatCard label="Jobs in this archetype" value={formatNumber(role.count)} detail={role.name} accent/>
        <StatCard label="Median annual salary" value={formatMoney(role.median_salary)} detail={`Full cohort ${formatMoney(data.metrics.median_salary)}`}/>
        <StatCard label="High-tier share" value={formatPercent(role.high_share)} detail="Frozen historical tier boundaries"/>
      </div>
      <div className="archetype-layout">
        <Panel title="The skill signature" description="Share of jobs mentioning each skill · profile vs full cohort" action={<Info label="About profile skill shares">Each share uses its own job population. A mention describes a posting, not proficiency.</Info>}>
          {signature.length ? <Chart option={skillOption} ariaLabel="Selected archetype skill shares compared with all saved jobs" height={420}/> : <EmptyState title="No extracted skill profile" description="These postings contain no matched skills in the saved vocabulary."/>}

        </Panel>
        <Panel title="Compensation within this profile" description={`${formatNumber(members.length)} unique jobs · ${range === 'central' ? '5th–95th percentile focus' : 'full annual salary range'}`} action={<Segmented label="Archetype salary range" value={range} onChange={setRange} options={[{value:'central',label:'Central 90%'},{value:'full',label:'Full range'}]}/>}>
          {members.length ? <Chart option={salaryOption} ariaLabel="Salary distribution within the selected archetype" height={300}/> : <EmptyState title="No salary observations"/>}
          <p className="subtle">Annual USD · every job counted once. {formatNumber(histogram.underflow)} below and {formatNumber(histogram.overflow)} above the displayed range. Summary statistics retain all jobs.</p>
        </Panel>
      </div>
      {technical && <Panel title="How distinct are the groups?" description="Discrete tested configurations · common original-space cosine silhouette · higher is better">
        <div className="archetype-layout">
          {summaries.length ? <Chart option={robustnessOption} ariaLabel="Bounded clustering robustness comparison by group count" height={Math.max(220,summaries.length*38+45)}/> : <EmptyState title="Robustness results unavailable" description="No new clustering is run in this app."/>}
          <div className="cluster-interpretation"><Badge tone="warning">Weak separation</Badge><h3>Useful profiles, overlapping jobs.</h3><p className="subtle">Separation remains weak. Stable seeds do not establish natural or externally validated job categories.</p><p className="subtle">Each bar is an actually tested configuration. No intermediate group counts are implied.</p></div>
        </div>
      </Panel>}
      <details className="detail-disclosure"><summary>{technical ? "Chart values, population & seed stability" : "Chart values & profile definitions"}</summary>
        {technical ? <p>Profiles retain saved K-Means assignments. Alternative SVD representations are robustness diagnostics, not an adopted classifier feature pipeline.</p> : <p>Profiles reflect saved groups of postings with similar extracted skill mentions. They are exploratory aids, not validated natural occupations or career recommendations.</p>}
        {technical && robustness && <p>Robustness sampled {formatNumber(robustness.provenance.sample_rows)} nonzero skill vectors with {robustness.configuration.seeds.length} seeds. Pairwise adjusted Rand index measures same-sample seed agreement, not external validation.</p>}
        <table><caption>Selected profile skill signature</caption><thead><tr><th scope="col">Skill</th><th scope="col">Archetype share</th><th scope="col">Full cohort share</th></tr></thead><tbody>{[...signature].reverse().map(skill => <tr key={skill.index}><th scope="row">{skill.skill}</th><td>{formatPercent(skill.share)}</td><td>{formatPercent(skill.overall_share)}</td></tr>)}</tbody></table>
        <table><caption>Selected profile salary distribution · displayed range in annual USD</caption><thead><tr><th scope="col">Range</th><th scope="col">Jobs</th></tr></thead><tbody>{histogram.counts.map((count,index) => <tr key={index}><th scope="row">{formatMoney(histogram.edges[index])} – {formatMoney(histogram.edges[index+1])}</th><td>{formatNumber(count)}</td></tr>)}</tbody></table>
        {technical && !!summaries.length && <table><caption>Actually tested robustness configurations</caption><thead><tr><th scope="col">Representation</th><th scope="col">k</th><th scope="col">Silhouette</th><th scope="col">Mean seed ARI</th><th scope="col">Cluster-share range</th></tr></thead><tbody>{summaries.map(row => <tr key={`${row.representation}-${row.k}`}><th scope="row">{representationName(row.representation)}</th><td>{row.k}</td><td>{formatScore(row.mean_cosine_silhouette_original)}</td><td>{formatScore(row.mean_pairwise_seed_ari)}</td><td>{formatPercent(row.min_cluster_share)} – {formatPercent(row.max_cluster_share)}</td></tr>)}</tbody></table>}
      </details>
    </>}
  </div>;
}
