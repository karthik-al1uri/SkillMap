import { useEffect, useMemo, useState } from 'react';
import type { EChartsOption } from 'echarts';
import type { ProductData, Rule } from '../data/types';
import { filterCohort, median, networkNeighbors } from '../data/logic';
import Chart, { chartBase, colors } from '../components/Chart';
import { Badge, EmptyState, PageHeading, Panel, Segmented, SearchSelect, Info, formatMoney, formatNumber, formatPercent } from '../components/UI';

type Scope = 'all' | 'technical';
const ruleName = (rule: Rule) => `${rule.antecedents} → ${rule.consequents}`;

export default function Skills({ data, technical = false }: { data: ProductData; technical?: boolean }) {
  const [compact, setCompact] = useState(() => window.matchMedia?.('(max-width: 560px)').matches ?? false);
  useEffect(() => {
    const media = window.matchMedia?.('(max-width: 560px)');
    if (!media) return;
    const update = () => setCompact(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  const columns = compact ? 3 : 4;
  const [scope, setScope] = useState<Scope>('all');
  const [graphVersion, setGraphVersion] = useState(0);
  const [measure, setMeasure] = useState('count');
  const [focusedId, setFocusedId] = useState<number | null>(null);
  const graph = data.networks[scope];
  const focused = graph.nodes.find(node => node.id === focusedId) ?? graph.nodes[0] ?? null;
  const neighbors = useMemo(() => focused ? networkNeighbors(graph, focused.id) : [], [graph, focused]);
  const connected = new Set([focused?.id, ...neighbors.map(neighbor => neighbor.node.id)]);
  const focusedMedian = useMemo(() => focused ? median(filterCohort(data.rows, { role: null, location: null, skill: focused.id }).map(row => row[0])) : null, [data.rows, focused]);
  const neighborBars = neighbors.slice(0, 8).reverse();
  const networkOption: EChartsOption = chartBase({
    tooltip: { trigger: 'item', renderMode: 'richText', formatter: params => {
      const event = params as unknown as { data: { skillId?: number; name?: string; value?: number; jaccard?: number }; dataType?: string };
      return event.dataType === 'edge' ? `${formatNumber(event.data.value)} shared jobs\nJaccard overlap: ${formatPercent(event.data.jaccard)}`
        : `${event.data.name}\n${formatNumber(event.data.value)} saved jobs\n${data.skills[event.data.skillId ?? -1]?.is_technical === true ? 'Technical skill · diamond' : data.skills[event.data.skillId ?? -1]?.is_technical === false ? 'Generic competency · circle' : 'Unclassified · rectangle'}`;
    } },
    grid: undefined, xAxis: undefined, yAxis: undefined,
    series: [{ type: 'graph', layout: 'none', roam: true, scaleLimit: { min: .5, max: 3 },
      left: compact ? 36 : 70, right: compact ? 36 : 70, top: 36, bottom: 65,
      data: graph.nodes.map((node, index) => ({ id: String(node.id), name: node.label, value: node.count, skillId: node.id,
        x: (index % columns) * 200, y: Math.floor(index / columns) * 140,
        symbol: data.skills[node.id]?.is_technical === true ? 'diamond' : data.skills[node.id]?.is_technical === false ? 'circle' : 'roundRect',
        symbolSize: 16 + 21 * Math.sqrt(node.count / Math.max(1, ...graph.nodes.map(item => item.count))),
        itemStyle: { color: node.id === focused?.id ? colors.accent : '#81769f',
          borderColor: node.id === focused?.id ? '#f1eaff' : '#2a2731', borderWidth: node.id === focused?.id ? 2 : 1,
          opacity: connected.has(node.id) ? 1 : .5 },
        label: { show: true, color: connected.has(node.id) ? colors.text : colors.muted, position: 'bottom', distance: 9, fontSize: 12, width: compact ? 82 : 125, overflow: 'break', lineHeight: 16 },
        labelLayout: { hideOverlap: true },
      })),
      links: graph.edges.map(edge => ({ source: String(edge.source), target: String(edge.target), value: edge.count, jaccard: edge.jaccard,
        lineStyle: { width: 1 + edge.jaccard * 6, color: '#8b7cad', opacity: edge.source === focused?.id || edge.target === focused?.id ? .65 : .055 } })),
      emphasis: { focus: 'adjacency', lineStyle: { opacity: 1, width: 3 }, itemStyle: { opacity: 1, borderWidth: 2 }, label: { color: colors.text } }, edgeSymbol: ['none', 'none'],
    }],
  });
  const neighborOption: EChartsOption = chartBase({
    xAxis: { type: 'value', minInterval: measure === 'count' ? 1 : undefined, axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'category', data: neighborBars.map(item => item.node.label), axisLabel: { color: colors.muted, width: 130, overflow: 'truncate' }, axisLine: { show: false }, axisTick: { show: false } },
    tooltip: { trigger: 'axis', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, valueFormatter: value => measure === 'count' ? `${formatNumber(Number(value))} shared jobs` : `${formatPercent(Number(value))} Jaccard overlap` },
    series: [{ type: 'bar', data: neighborBars.map(item => measure === 'count' ? item.count : item.jaccard), barMaxWidth: 20, itemStyle: { color: colors.accent, borderRadius: [0, 4, 4, 0] } }],
  });
  const rules = [...(scope === 'technical' ? data.rules.technical ?? [] : data.rules.general ?? [])]
    .sort((a, b) => b.lift - a.lift || b.support - a.support || ruleName(a).localeCompare(ruleName(b))).slice(0, 10);
  const highRules = [...(data.rules.high ?? [])].sort((a, b) => (b.high_tier_rate ?? -1) - (a.high_tier_rate ?? -1) || b.lift - a.lift).slice(0, 5);

  return <div className="page skills-page">
    <PageHeading eyebrow="RELATIONSHIPS, REVEALED" title="Skills rarely work alone." description="Follow the capabilities that appear together in saved job postings."><Badge>Observed co-occurrence</Badge></PageHeading>
    <div className="toolbar">
      <Segmented label="Skill network scope" value={scope} options={[{ value: 'all', label: 'All skills' }, { value: 'technical', label: 'Technical' }]} onChange={value => { setScope(value as Scope); setFocusedId(null); }} />
      <SearchSelect label="Inspect a skill" value={focused ? String(focused.id) : ''} onChange={value => setFocusedId(Number(value))} options={[...graph.nodes].sort((a,b) => a.label.localeCompare(b.label)).map(node => ({value:String(node.id),label:node.label}))}/>
      <button type="button" className="ghost-button" onClick={() => { setFocusedId(null); setGraphVersion(value => value + 1); }}>Reset network</button>
      <span className="subtle">{formatNumber(graph.nodes.length)} skills · {formatNumber(graph.edges.length)} displayed links</span>
    </div>
    <div className="network-layout">
      <Panel title="The skill constellation" description="Click to focus · drag to pan · scroll to zoom" action={<Info label="How to read the network"><p>Links count shared postings; width reflects Jaccard overlap. This is a descriptive neighborhood, not a validated community map. Positions serve legibility, not inferred similarity.</p></Info>}>
        {graph.nodes.length ? <Chart option={networkOption} ariaLabel="Observed skill co-occurrence network" key={`${scope}-${graphVersion}`} height={Math.max(430, Math.ceil(graph.nodes.length / columns) * 90)} onClick={event => {
          const item = event.data as { skillId?: number } | undefined;
          if (typeof item?.skillId === 'number') setFocusedId(item.skillId);
        }} /> : <EmptyState title="No observed network" description="No saved skills meet this neighborhood's definition." />}
        <div className="chart-key"><span>◇ Technical</span><span>○ Generic</span><span>▢ Unclassified</span></div>

      </Panel>
      <Panel title={focused ? `Alongside ${focused.label}` : 'A closer connection'} description={focused ? `${formatNumber(focused.count)} saved jobs · median annual salary ${formatMoney(focusedMedian)}` : 'Choose a skill to inspect its connections'} action={<Segmented label="Neighbor measure" value={measure} onChange={setMeasure} options={[{value:'count',label:'Shared jobs'},{value:'jaccard',label:'Jaccard'}]}/>}>
        {neighborBars.length ? <Chart option={neighborOption} ariaLabel="Shared-posting counts for the selected skill's neighbors" height={280} /> : <EmptyState title="No displayed neighbors" description="Try another skill or the broader network." />}
        <div className="neighbor-links" aria-label="Related skills">{neighbors.slice(0, 5).map(item => <button key={item.node.id} type="button" onClick={() => setFocusedId(item.node.id)}><span>{item.node.label}</span><span>{formatNumber(item.count)} jobs · {formatPercent(item.jaccard)} overlap →</span></button>)}</div>

      </Panel>
    </div>
    <details className="detail-disclosure"><summary>Selected skill connections · accessible values</summary><table><thead><tr><th>Connected skill</th><th>Shared jobs</th><th>Jaccard overlap</th></tr></thead><tbody>{neighbors.map(item => <tr key={item.node.id}><th>{item.node.label}</th><td>{formatNumber(item.count)}</td><td>{formatPercent(item.jaccard)}</td></tr>)}</tbody></table></details>
    {technical && <details className="detail-disclosure"><summary>Inspect the strongest saved association rules</summary>
      <p className="subtle">{scope === 'technical' ? 'Every skill in these rules is technical; mining support is at least 2%.' : 'General mining support is at least 5%.'} Confidence must exceed 0.5 and lift must exceed 1.5. These saved results are not remined by the app.</p>
      {rules.length ? <table><thead><tr><th scope="col">Rule</th><th scope="col">Lift</th><th scope="col">Support</th><th scope="col">Confidence</th></tr></thead><tbody>{rules.map(rule => <tr key={ruleName(rule)}><th scope="row">{ruleName(rule)}</th><td>{rule.lift.toFixed(2)}</td><td>{formatPercent(rule.support)}</td><td>{formatPercent(rule.confidence)}</td></tr>)}</tbody></table> : <EmptyState title={data.rules[scope === 'technical' ? 'technical' : 'general'] == null ? 'Saved rule artifacts unavailable' : 'No saved rules in this view'} description="Observed network links do not imply that strict association rules qualify." />}
      {!!highRules.length && <><h3>High-tier view · top five by High-tier share</h3><p className="subtle">Mining support uses High-tier jobs. High-tier share uses all jobs containing the complete itemset.</p><table><thead><tr><th scope="col">Rule</th><th scope="col">High-tier share</th><th scope="col">Jobs with itemset</th></tr></thead><tbody>{highRules.map(rule => <tr key={ruleName(rule)}><th scope="row">{ruleName(rule)}</th><td>{formatPercent(rule.high_tier_rate)}</td><td>{formatNumber(rule.jobs_with_itemset)}</td></tr>)}</tbody></table></>}
    </details>}
  </div>;
}
