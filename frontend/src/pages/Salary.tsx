import { useMemo, useState } from 'react';
import type { EChartsOption } from 'echarts';
import type { ProductData } from '../data/types';
import { filterCohort, groupSalary, median, salaryHistogram } from '../data/logic';
import Chart, { chartBase, colors } from '../components/Chart';
import { Badge, EmptyState, PageHeading, Panel, SearchSelect, Segmented, Select, StatCard, formatMoney, formatNumber, formatPercent } from '../components/UI';

type Dimension = 'skill' | 'role' | 'location';
/** Linear interpolation of the observed sorted salaries; does not change cohort membership. */
function percentile(values: readonly number[], share: number): number | null {
  if (!values.length) return null;
  const index = (values.length - 1) * share;
  const lower = Math.floor(index);
  return values[lower] + (values[Math.ceil(index)] - values[lower]) * (index - lower);
}

export default function Salary({ data }: { data: ProductData }) {
  const [role, setRole] = useState('');
  const [location, setLocation] = useState('');
  const [skill, setSkill] = useState('');
  const [minimum, setMinimum] = useState('25');
  const [dimension, setDimension] = useState<Dimension>('skill');
  const [range, setRange] = useState('central');
  const [focused, setFocused] = useState<number | null>(null);
  const cohort = useMemo(() => filterCohort(data.rows, {
    role: role === '' ? null : Number(role), location: location === '' ? null : Number(location), skill: skill === '' ? null : Number(skill),
  }), [data.rows, role, location, skill]);
  const sorted = useMemo(() => cohort.map(row => row[0]).sort((a, b) => a - b), [cohort]);
  const cohortMedian = useMemo(() => median(sorted), [sorted]);
  const highShare = cohort.length ? cohort.filter(row => row[4] === 2).length / cohort.length : null;
  const groups = useMemo(() => groupSalary(cohort, dimension, Number(minimum))
    .sort((a, b) => b.count - a.count || a.index - b.index).slice(0, 12)
    .sort((a, b) => a.median_salary - b.median_salary || a.index - b.index), [cohort, dimension, minimum]);
  const p5 = percentile(sorted, .05);
  const p95 = percentile(sorted, .95);
  const histogram = useMemo(() => salaryHistogram(cohort, 24,
    range === 'central' && p5 !== null && p95 !== null && p5 < p95 ? [p5, p95] : undefined), [cohort, range, p5, p95]);
  const name = (index: number) => dimension === 'skill' ? data.skills[index]?.name ?? 'Unknown skill'
    : dimension === 'role' ? data.roles[index]?.name ?? 'Unknown archetype' : data.locations[index] ?? 'Unrecorded location';
  const selected = groups.find(group => group.index === focused);
  const comparison: EChartsOption = chartBase({
    grid: { left: 8, right: 88, top: 24, bottom: 20, containLabel: true },
    tooltip: { trigger: 'item', renderMode: 'richText', confine: true, formatter: event => {
      const item = Array.isArray(event) ? event[0] : event;
      const group = groups[item.dataIndex];
      return group ? `${name(group.index)}\nMedian ${formatMoney(group.median_salary)}\n${formatNumber(group.count)} unique jobs · High-tier ${formatPercent(group.high_share)}` : '';
    } },
    xAxis: { type: 'value', axisLabel: { color: colors.muted, formatter: value => formatMoney(Number(value)) }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'category', data: groups.map(group => name(group.index)), axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { color: colors.text, fontSize: 12, width: 190, overflow: 'break', lineHeight: 16 } },
    series: [{ type: 'scatter', symbolSize: 13, label: { show: true, position: 'right', distance: 9, color: colors.muted, fontSize: 11, formatter: event => `n=${formatNumber(groups[event.dataIndex]?.count)}` }, data: groups.map((group, index) => ({ value: [group.median_salary, index], groupIndex: group.index,
      name: name(group.index), itemStyle: { color: focused === group.index ? '#eee6ff' : colors.accent, opacity: focused === null || focused === group.index ? 1 : .35 } })),
      emphasis: { scale: 1.4 },
      markLine: cohortMedian === null ? undefined : { silent: true, symbol: 'none', lineStyle: { color: colors.reference, type: 'dashed' }, label: { formatter: 'Cohort median', color: colors.muted, position: 'insideEndTop' }, data: [{ xAxis: cohortMedian }] },
    }],
  });
  const distribution: EChartsOption = chartBase({
    grid: { left: 10, right: 16, top: 24, bottom: 24, containLabel: true },
    tooltip: { trigger: 'item', renderMode: 'richText', formatter: event => { const item = Array.isArray(event) ? event[0] : event; return `${item.name}\n${formatNumber(Number(item.value))} unique jobs`; } },
    xAxis: { type: 'category', data: histogram.counts.map((_, i) => `${formatMoney(histogram.edges[i])} – ${formatMoney(histogram.edges[i + 1])}`),
      axisTick: { show: false }, axisLine: { lineStyle: { color: colors.grid } },
      axisLabel: { color: colors.muted, fontSize: 12, hideOverlap: true, interval: 5, formatter: (_value, index) => formatMoney(histogram.edges[index]) } },
    yAxis: { type: 'value', minInterval: 1, name: 'Jobs', nameTextStyle: { color: colors.muted }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } }, axisLabel: { color: colors.muted } },
    series: [{ type: 'bar', data: histogram.counts, barCategoryGap: '4%', itemStyle: { color: colors.accent, opacity: .8, borderRadius: [3, 3, 0, 0] }, emphasis: { itemStyle: { opacity: 1 } } }],
  });
  const reset = () => { setRole(''); setLocation(''); setSkill(''); setMinimum('25'); setFocused(null); setRange('central'); };
  const focusComparison = (index: number) => setFocused(current => current === index ? null : index);
  const drillInto = () => {
    if (focused === null) return;
    if (dimension === 'skill') setSkill(String(focused));
    else if (dimension === 'role') setRole(String(focused));
    else setLocation(String(focused));
    setFocused(null);
  };

  return <div className="page salary-page">
    <PageHeading eyebrow="COMPENSATION EXPLORER" title="Put a number to the picture." description="Find meaningful pay differences within the saved job cohort."><Badge>Historical · unique jobs</Badge></PageHeading>
    <div className="filter-toolbar" aria-label="Salary cohort filters">
      <SearchSelect label="Archetype" value={role} onChange={value => { setRole(value); setFocused(null); }} options={[{ value: '', label: 'All archetypes' }, ...data.roles.map((item, index) => ({ value: String(index), label: item.name }))]}/>
      <SearchSelect label="Recorded location" value={location} onChange={value => { setLocation(value); setFocused(null); }} options={[{ value: '', label: 'All locations' }, ...data.locations.map((item, index) => ({ value: String(index), label: item ?? 'Unrecorded location' }))]}/>
      <SearchSelect label="Skill mentioned" value={skill} onChange={value => { setSkill(value); setFocused(null); }} options={[{ value: '', label: 'Any skill' }, ...[...data.skills].sort((a,b) => a.name.localeCompare(b.name)).map(item => ({ value: String(item.index), label: item.name }))]}/>
      <Select label="Minimum group size" value={minimum} onChange={value => { setMinimum(value); setFocused(null); }}>{[10, 25, 50].map(value => <option key={value} value={value}>{value} jobs</option>)}</Select>
      <button type="button" className="text-button" onClick={reset}>Reset filters</button>
    </div>
    <div className="salary-layout">
      <Panel className="salary-comparison" title="Where median pay differs" description="The 12 most represented eligible groups · select a dot to inspect" action={<Segmented label="Compare salary by" value={dimension} options={[{ value: 'skill', label: 'Skill' }, { value: 'role', label: 'Archetype' }, { value: 'location', label: 'Location' }]} onChange={value => { setDimension(value as Dimension); setFocused(null); }} />}>
        {groups.length ? <Chart option={comparison} ariaLabel="Median annual salary by selected comparison group" height={Math.max(340, groups.length * 42 + 65)} onClick={event => { const item = event.data as { groupIndex?: number } | undefined; if (typeof item?.groupIndex === 'number') focusComparison(item.groupIndex); }}/> : <EmptyState title="No comparison groups" description={`No groups have at least ${minimum} jobs. Lower the minimum or broaden the filters.`} />}
        <div className="salary-focus" aria-live="polite">{selected ? <><strong>{name(selected.index)}</strong><span>{formatMoney(selected.median_salary)} median · {formatNumber(selected.count)} jobs</span><button className="ghost-button" type="button" onClick={drillInto}>Filter to this group →</button><button className="text-button" type="button" onClick={() => setFocused(null)}>Clear selection</button></> : <span className="subtle">Each dot is a group median. The dashed line is the selected cohort median.</span>}</div>
      </Panel>
      <aside className="salary-summary" aria-label="Selected salary cohort" aria-live="polite">
        <StatCard label="Median annual salary" value={formatMoney(cohortMedian)} detail="Your selected cohort · annual USD" accent />
        <StatCard label="Jobs in your selection" value={formatNumber(cohort.length)} detail="Every matching job counted once" />
        <div className="salary-context"><StatCard label="High-tier share" value={formatPercent(highShare)} detail="Frozen historical boundaries"/><StatCard label="Full-cohort median" value={formatMoney(data.metrics.median_salary)} detail="Unfiltered reference"/></div>
        <p className="subtle">Observed differences are descriptive, not causal salary premiums. Skill groups can overlap.</p>
      </aside>
      <Panel className="salary-distribution" title="The shape of observed pay" description={range === 'central' ? 'Focus on the 5th–95th percentiles' : 'Every salary, including outliers'} action={<Segmented label="Salary distribution range" value={range} onChange={setRange} options={[{value:'central', label:'Central 90%'},{value:'full', label:'Full range'}]}/>}>
        {cohort.length ? <Chart option={distribution} ariaLabel="Salary distribution of the filtered unique-job cohort" height={255}/> : <EmptyState title="No matching jobs" description="Clear a filter to explore a broader cohort." />}
        {!!cohort.length && <p className="range-note">{range === 'central' ? `${formatNumber(histogram.counts.reduce((sum,count) => sum+count,0))} jobs shown · ${formatNumber(histogram.underflow)} below P5 · ${formatNumber(histogram.overflow)} above P95. ` : `${formatNumber(cohort.length)} jobs shown · no omitted outliers. `}All summary statistics always use the full selected cohort.</p>}
      </Panel>
    </div>
    <details className="detail-disclosure"><summary>Accessible values & keyboard drilldown</summary><table><caption>Selected salary groups · choose a group to highlight its dot</caption><thead><tr><th>Group</th><th>Jobs</th><th>Median annual salary</th><th>Focus</th></tr></thead><tbody>{groups.map(group => <tr key={group.index}><th>{name(group.index)}</th><td>{formatNumber(group.count)}</td><td>{formatMoney(group.median_salary)}</td><td><button type="button" className="text-button" aria-pressed={focused === group.index} onClick={() => focusComparison(group.index)}>Inspect {name(group.index)}</button></td></tr>)}</tbody></table><table><caption>Selected cohort distribution</caption><thead><tr><th>Annual salary range</th><th>Jobs</th></tr></thead><tbody>{histogram.counts.map((count,index) => <tr key={index}><th>{formatMoney(histogram.edges[index])} – {formatMoney(histogram.edges[index+1])}</th><td>{formatNumber(count)}</td></tr>)}</tbody></table><p>Outside the displayed bins: {formatNumber(histogram.underflow)} below and {formatNumber(histogram.overflow)} above. Minimum group size applies only to the comparison plot.</p></details>
  </div>;
}
