import { useMemo, useState } from 'react';
import type { EChartsOption } from 'echarts';
import { normalizeConfusion } from '../data/logic';
import type { FixedRule, ProductData } from '../data/types';
import Chart, { chartBase, colors } from '../components/Chart';
import { Badge, EmptyState, PageHeading, Panel, Segmented, Select, StatCard, formatNumber, formatPercent, formatScore } from '../components/UI';

type Source = 'adzuna' | 'muse';
const numeric = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;

export default function Live({ data }: { data: ProductData }) {
  const [matrixMode, setMatrixMode] = useState('share');
  const [cell, setCell] = useState('');
  const [source, setSource] = useState<Source>('adzuna');
  const [cap, setCap] = useState('all');
  const [selection, setSelection] = useState<string | null>(null);
  const report = data.live.evaluation;
  const dates = data.metrics.live_snapshot_dates?.length ? data.metrics.live_snapshot_dates : report?.source_status.fetch_date ? [report.source_status.fetch_date] : [];
  const capped = cap === 'capped';
  const fixedRules = (capped ? data.live.fixed_rules_capped : data.live.fixed_rules_uncapped) ?? [];
  const selected = fixedRules.find(rule => rule.rule === selection) ?? null;
  const comparison = useMemo(() => [...(data.live.comparison ?? [])].sort((a, b) => b.muse_share - a.muse_share).slice(0, 10).reverse(), [data.live.comparison]);
  const measurable = fixedRules.filter(rule => rule.support !== null && rule.lift !== null);
  const a = report?.adzuna;
  const confusion = a?.confusion_rows_actual_columns_predicted ?? [];
  const normalized = normalizeConfusion(confusion);
  const liveLabels = a?.labels ?? data.tiers;
  const precision = data.tiers.map(tier => {
    const index = liveLabels.indexOf(tier);
    const predicted = confusion.reduce((sum,row) => sum + (row[index] ?? 0), 0);
    return predicted ? (confusion[index]?.[index] ?? 0) / predicted : null;
  });
  const cellParts = cell.split('-').map(Number);
  const selectedCell = cell && confusion[cellParts[0]] ? { observed: cellParts[0], predicted: cellParts[1], count: confusion[cellParts[0]][cellParts[1]] } : null;
  const m = capped ? report?.muse_company_capped : report?.muse_uncapped;
  const contract = report?.predeclared_contract;
  const verticalAxes = {
    xAxis: { type: 'category' as const, data: data.tiers, axisLabel: { color: colors.muted }, axisLine: { show: false }, axisTick: { show: false } },
    yAxis: { type: 'value' as const, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' as const } }, axisLabel: { color: colors.muted } },
  };
  const legend = { top: 0, left: 0, textStyle: { color: colors.muted, fontSize: 12 }, icon: 'roundRect', itemWidth: 14, itemHeight: 11 };
  const adzunaMix: EChartsOption = chartBase({ ...verticalAxes, legend, grid: { left: 10, right: 15, top: 45, bottom: 14, containLabel: true },
    series: a ? [{ type: 'bar', name: 'Observed', data: data.tiers.map(tier => a.actual_counts[tier]), barMaxWidth: 35, itemStyle: { color: colors.reference, borderRadius: [4, 4, 0, 0] } },
      { type: 'bar', name: 'Predicted', data: data.tiers.map(tier => a.predicted_counts[tier]), barMaxWidth: 35, itemStyle: { color: colors.accent, borderRadius: [4, 4, 0, 0] } }] : [],
  });
  const adzunaClasses: EChartsOption = chartBase({ ...verticalAxes, legend, grid: { left: 10, right: 15, top: 45, bottom: 14, containLabel: true },
    yAxis: { ...verticalAxes.yAxis, min: 0, max: 1, axisLabel: { color: colors.muted, formatter: (value: number) => formatPercent(Number(value)) } },
    tooltip: { trigger: 'axis', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, valueFormatter: value => formatPercent(Number(value)) },
    series: a ? [{ type: 'bar', name: 'Precision · derived', data: precision.map((value,index) => ({value,itemStyle:{color:colors.tiers[index],decal:{symbol:'circle',dashArrayX:[1,2],dashArrayY:[2,4],color:'#0005'}}})), barMaxWidth: 25 }, { type: 'bar', name: 'Recall', data: data.tiers.map((tier,index) => ({value:a.per_class_recall[tier],itemStyle:{color:colors.tiers[index]}})), barMaxWidth: 35, itemStyle: { color: colors.accent, borderRadius: [4, 4, 0, 0] } },
      { type: 'bar', name: 'F1', data: data.tiers.map((tier,index) => ({value:a.per_class_f1[tier],itemStyle:{color:colors.tiers[index],decal:{symbol:'rect',dashArrayX:[1,0],dashArrayY:[2,4],rotation:-Math.PI/4,color:'#0005'}}})), barMaxWidth: 35, itemStyle: { color: '#71638f', borderRadius: [4, 4, 0, 0] } }] : [],
  });
  const museShares: EChartsOption = chartBase({ legend, grid: { left: 8, right: 20, top: 42, bottom: 14, containLabel: true },
    xAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: colors.muted, formatter: (value: number) => formatPercent(Number(value)) }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'category', data: comparison.map(row => row.skill), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: colors.muted, width: 130, overflow: 'truncate' } },
    tooltip: { trigger: 'item', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, valueFormatter: value => formatPercent(Array.isArray(value) ? Number(value[0]) : Number(value)) },
    series: [{ type: 'scatter', name: 'Training tech · diamond', symbol: 'diamond', symbolSize: 11, data: comparison.map((row, index) => ({ value: [row.train_tech_share, index], name: row.skill })), itemStyle: { color: colors.reference } },
      { type: 'scatter', name: capped ? 'Muse capped · circle' : 'Muse · circle', symbol: 'circle', symbolSize: 13, data: comparison.map((row, index) => ({ value: [capped ? row.muse_share_capped : row.muse_share, index], name: row.skill })), itemStyle: { color: colors.accent } }],
  });
  const museRules: EChartsOption = chartBase({
    grid: { left: 12, right: 18, top: 25, bottom: 32, containLabel: true },
    xAxis: { type: 'value', name: 'Support', nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: colors.muted }, axisLabel: { color: colors.muted, formatter: (value: number) => formatPercent(Number(value)) }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    yAxis: { type: 'value', name: 'Lift', nameTextStyle: { color: colors.muted }, axisLabel: { color: colors.muted }, splitLine: { lineStyle: { color: colors.grid, type: 'dashed' } } },
    tooltip: { trigger: 'item', renderMode: 'richText', backgroundColor: '#242731', borderColor: '#3a3d47', textStyle: { color: colors.text }, formatter: params => {
      const event = params as unknown as {data:{ruleIndex?:number}};
      const rule = typeof event.data.ruleIndex === 'number' ? fixedRules[event.data.ruleIndex] : null;
      return rule ? `${rule.rule}\nSupport ${formatPercent(rule.support)} · Confidence ${formatPercent(rule.confidence)}\nLift ${formatScore(rule.lift)}\n${rule.passes_primary_005 ? 'Meets primary thresholds' : 'Does not meet primary thresholds'}` : '';
    } },
    series: [{ type: 'scatter', data: measurable.map(rule => ({ value: [rule.support!, rule.lift!], name: rule.rule, ruleIndex: fixedRules.indexOf(rule),
      symbol: rule.passes_primary_005 ? 'diamond' : 'circle', symbolSize: rule.rule === selection ? 23 : rule.passes_primary_005 ? 15 : 10, itemStyle: { color: rule.passes_primary_005 ? colors.accent : colors.reference, borderColor: rule.rule === selection ? colors.text : 'transparent', borderWidth: 2 } })),
      markLine: contract ? { silent: true, symbol: 'none', label: { color: colors.muted }, lineStyle: { color: '#7d738d', type: 'dashed', width: 1 },
        data: [{ xAxis: contract.primary_support_inclusive, name: 'Primary support' }, { yAxis: contract.lift_strictly_greater_than, name: 'Lift threshold' }] } : undefined,
    }],
  });
  const ruleStatus = (rule: FixedRule) => rule.support === null || rule.confidence === null || rule.lift === null ? 'Not measurable' : rule.passes_primary_005 ? 'Confirmed' : 'Not confirmed';

  return <div className="page live-page admin-page">
    <PageHeading eyebrow="CACHED CROSS-SOURCE EVIDENCE" title="A reality check, with context." description="Explore how saved historical evidence travels across two different source populations."><Badge>{dates.length ? `Cached · ${dates.join(' · ')}` : 'Date unavailable'}</Badge></PageHeading>
    <div className="source-strip"><div><span className="source-tag">ADZUNA</span><strong>{report?.source_status.fetch_date ?? 'Date unavailable'}</strong><small>{a ? `${formatNumber(a.n)} cached salary-labeled predictions` : 'Cached predictions unavailable'}</small></div><div><span className="source-tag">MUSE</span><strong>{Array.isArray(report?.source_status.available_muse_dates) ? (report.source_status.available_muse_dates as string[]).join(' · ') : dates.join(' · ') || 'Date unavailable'}</strong><small>{m ? `${formatNumber(m.eligible_postings)} eligible skill baskets · ${capped ? 'employer capped' : 'uncapped'}` : 'Cached baskets unavailable'}</small></div><p>Two source populations.<br/>One saved snapshot, no temporal trend.</p></div>
    <div className="toolbar">
      <Segmented label="Cached live source" value={source} options={[{ value: 'adzuna', label: 'Adzuna' }, { value: 'muse', label: 'Muse' }]} onChange={value => { setSource(value as Source); setSelection(null); }} />
      {source === 'muse' && <Select label="Employer sensitivity" value={cap} onChange={setCap}><option value="all">All cached employers</option><option value="capped">Cap 25 per employer</option></Select>}
      <span className="subtle">Saved snapshots only · no new requests</span>
    </div>
    {!report ? <EmptyState title="Cached validation unavailable" description="No historical model score is substituted for a missing cross-source evaluation." /> : <>
      <div className={`live-evidence ${source === 'muse' ? 'is-muse' : ''}`}>
      {source === 'adzuna' && a && <Panel title="Does the model beat a simple benchmark?" description="Accuracy on the same cached Adzuna population">
        <Chart ariaLabel="Cached accuracy versus majority benchmarks" height={245} option={chartBase({grid:{left:8,right:68,top:12,bottom:20,containLabel:true},xAxis:{type:'value',min:0,max:1,axisLabel:{formatter:value=>formatPercent(Number(value))}},yAxis:{type:'category',data:['Historical training-majority','Observed-majority oracle','Model predictions'],axisLabel:{color:colors.text,fontSize:12,width:190,overflow:'break',lineHeight:17}},tooltip:{trigger:'axis',valueFormatter:value=>formatPercent(Number(value))},series:[{type:'bar',barMaxWidth:22,data:[{value:a.comparators.historical_training_majority.accuracy,itemStyle:{color:colors.reference}},{value:a.comparators.observed_majority_oracle.accuracy,itemStyle:{color:colors.tiers[1],decal:{symbol:'rect',dashArrayX:[1,0],dashArrayY:[2,4],color:'#0005'}}},{value:a.accuracy,itemStyle:{color:colors.accent}}],label:{show:true,position:'right',color:colors.text,formatter:event=>formatPercent(event.value == null ? null : Number(event.value))}}]})}/>
        <p className="subtle">The oracle uses observed evaluation labels; it is a diagnostic reference, not a training-only baseline.</p>
      </Panel>}
      <div className="live-summary">
        {source === 'adzuna' && a ? <>
          <StatCard label="Cached predictions" value={formatNumber(a.n)} detail="Same observed and predicted population" accent />
          <StatCard label="Accuracy" value={formatScore(a.accuracy)} detail={`95% interval ${formatScore(a.bootstrap.accuracy.low)}–${formatScore(a.bootstrap.accuracy.high)}`} />
          <StatCard label="Macro F1" value={formatScore(a.macro_f1)} detail="Equal weight for each salary tier" />
          <StatCard label="Observed-majority oracle" value={formatScore(a.comparators.observed_majority_oracle.accuracy)} detail="Diagnostic accuracy · uses evaluation labels" />
        </> : source === 'muse' && m ? <>
          <StatCard label="Eligible cached postings" value={formatNumber(m.eligible_postings)} detail={`${formatNumber(m.all_postings)} before skill eligibility`} accent />
          <StatCard label="Fixed rules confirmed" value={`${m.primary_confirmed} / ${m.fixed_rules}`} detail="Primary frozen thresholds" />
          <StatCard label="Eligible employers" value={formatNumber(m.employers_eligible)} detail={capped ? 'At most 25 postings per employer' : 'All cached employers'} />
          <StatCard label="Support sensitivity" value={`${m.sensitivity_confirmed} / ${m.fixed_rules}`} detail="Separate 2% support diagnostic" />
        </> : null}
      </div>
      </div>
      <div className="live-layout">
        <Panel title={source === 'adzuna' ? 'The observed vs predicted mix' : 'Same skills, different populations'} description={source === 'adzuna' ? 'Postings by salary tier · same cached population' : 'Muse vs training-tech skill shares · a fixed set of skills'}>
          {source === 'adzuna' ? a ? <Chart option={adzunaMix} ariaLabel="Cached Adzuna observed and predicted salary-tier counts" height={330} /> : <EmptyState title="Cached predictions unavailable" />
            : comparison.length ? <Chart option={museShares} ariaLabel="Cached Muse and historical training-tech skill shares" height={330} /> : <EmptyState title="Skill-share comparison unavailable" />}
          <p className="subtle">{source === 'adzuna' ? 'Both bars use the same set of cached predictions; model-training rows are a different population.' : 'Descriptions, source coverage, and employer mix differ. These shares cannot establish a temporal demand trend.'}</p>
        </Panel>
        <Panel title={source === 'adzuna' ? 'Performance across salary tiers' : 'Which historical rules transfer?'} description={source === 'adzuna' ? 'Precision = dotted · recall = solid · F1 = hatched' : 'Support and lift · select a fixed rule to inspect it'}>
          {source === 'adzuna' ? a ? <Chart option={adzunaClasses} ariaLabel="Cached Adzuna recall and F1 by salary tier" height={330} /> : <EmptyState title="Per-tier results unavailable" />
            : measurable.length ? <Chart option={museRules} ariaLabel="Fixed-rule support and lift on cached Muse postings" height={330} onClick={event => {
              const item = event.data as { ruleIndex?: number } | undefined;
              if (typeof item?.ruleIndex === 'number' && fixedRules[item.ruleIndex]) setSelection(fixedRules[item.ruleIndex].rule);
            }} /> : <EmptyState title="No measurable fixed rules" description="Undefined rule metrics are retained in the table below." />}
          {source === 'muse' && !!fixedRules.length && <Select label="Inspect a fixed rule" value={selection ?? ''} onChange={setSelection}><option value="">Choose a saved rule</option>{fixedRules.map(rule => <option key={rule.rule} value={rule.rule}>{rule.rule}</option>)}</Select>}
          {source === 'muse' && <p className="subtle" aria-live="polite">{selected ? <><strong>{selected.rule}</strong><br />Confidence {formatPercent(selected.confidence)} · support interval {formatPercent(numeric(selected.support_ci_low))}–{formatPercent(numeric(selected.support_ci_high))} · bootstrap pass fraction {formatPercent(numeric(selected.bootstrap_pass_fraction_primary_005))}</> : 'Select a rule for uncertainty. Diamonds meet primary thresholds; circles do not.'}</p>}
        </Panel>
      </div>
      {source === 'adzuna' && a && <Panel className="validation-matrix" title="Cached Adzuna confusion matrix" description="Observed rows → predicted columns · same cached evaluation population" action={<Segmented label="Cached confusion matrix units" value={matrixMode} onChange={setMatrixMode} options={[{value:'counts',label:'Counts'},{value:'share',label:'Row share'}]}/>}>
        <div className="admin-mining-layout"><Chart ariaLabel="Cached Adzuna confusion matrix" height={360} onClick={event=>{const value=(event.data as {value?:number[]}|undefined)?.value;if(value)setCell(`${value[1]}-${value[0]}`);}} option={chartBase({visualMap:{show:false,min:0,max:1,dimension:3,inRange:{color:['#1e2230','#b9a3ff']}},grid:{left:55,right:25,top:15,bottom:35},xAxis:{type:'category',data:liveLabels,axisLabel:{color:colors.text,fontSize:14}},yAxis:{type:'category',data:liveLabels,inverse:true,axisLabel:{color:colors.text,fontSize:14}},tooltip:{trigger:'item',renderMode:'richText',formatter:params=>{const event=params as unknown as {value:number[]};const [x,y,count,share]=event.value;return `Observed ${liveLabels[y]} → Predicted ${liveLabels[x]}\n${formatNumber(confusion[y][x])} jobs · ${formatPercent(share)} of observed tier`; }},series:[{type:'heatmap',data:confusion.flatMap((row,y)=>row.map((count,x)=>({value:[x,y,matrixMode==='share'?normalized[y][x]:count,normalized[y][x]],itemStyle:{borderColor:cell===`${y}-${x}`?colors.tiers[1]:'#11151f',borderWidth:cell===`${y}-${x}`?3:2},label:{color:normalized[y][x]>.65?'#151121':colors.text}}))),label:{show:true,fontSize:18,color:colors.text,formatter:event=>{const value=(event.value as number[])[2];return matrixMode==='share'?formatPercent(value):formatNumber(value);}}}]})}/><div className="admin-rule-detail"><Select label="Inspect confusion cell" value={cell} onChange={setCell}><option value="">Choose observed → predicted</option>{liveLabels.flatMap((observed,y)=>liveLabels.map((predicted,x)=><option key={`${y}-${x}`} value={`${y}-${x}`}>{observed} → {predicted}</option>))}</Select><p aria-live="polite">{selectedCell?`${formatNumber(selectedCell.count)} jobs observed as ${liveLabels[selectedCell.observed]} and predicted as ${liveLabels[selectedCell.predicted]}. ${formatPercent(normalized[selectedCell.observed][selectedCell.predicted])} of that observed tier.`:'Select a cell to inspect its count and observed-tier share.'}</p><p className="subtle">Per-class precision is derived from these saved counts: diagonal / predicted column total. Precision is unavailable when a tier has no predictions.</p></div></div>
      </Panel>}
      {source === 'muse' && report && <Panel title="Employer concentration sensitivity" description="The company cap is applied before the at-least-two-skill eligibility filter"><table><thead><tr><th>Population</th><th>All postings</th><th>Eligible</th><th>Employers</th><th>Primary confirmations</th><th>2% support diagnostic</th></tr></thead><tbody>{[[report.muse_uncapped,'Uncapped'],[report.muse_company_capped,'At most 25 per company']].map(([value,label])=>{const summary=value as typeof report.muse_uncapped;return <tr key={String(label)}><th>{String(label)}</th><td>{formatNumber(summary.all_postings)}</td><td>{formatNumber(summary.eligible_postings)}</td><td>{formatNumber(summary.employers_eligible)}</td><td>{summary.primary_confirmed} / {summary.fixed_rules}</td><td>{summary.sensitivity_confirmed} / {summary.fixed_rules}</td></tr>;})}</tbody></table></Panel>}
      <details className="detail-disclosure"><summary>Chart values, baselines & population definitions</summary>
        {source === 'adzuna' && a && <table><caption>Cached Adzuna · same prediction population</caption><thead><tr><th>Tier</th><th>Observed</th><th>Predicted</th><th>Precision · derived</th><th>Recall</th><th>F1</th></tr></thead><tbody>{data.tiers.map((tier,index)=><tr key={tier}><th>{tier}</th><td>{formatNumber(a.actual_counts[tier])}</td><td>{formatNumber(a.predicted_counts[tier])}</td><td>{formatPercent(precision[index])}</td><td>{formatPercent(a.per_class_recall[tier])}</td><td>{formatPercent(a.per_class_f1[tier])}</td></tr>)}</tbody></table>}
        {source === 'adzuna' && a && <table><caption>Confusion counts · observed rows, predicted columns</caption><thead><tr><th>Observed</th>{liveLabels.map(tier=><th key={tier}>{tier}</th>)}</tr></thead><tbody>{confusion.map((row,index)=><tr key={index}><th>{liveLabels[index]}</th>{row.map((value,column)=><td key={column}>{formatNumber(value)}</td>)}</tr>)}</tbody></table>}
        {source === 'muse' && <table><caption>Skill share comparison · distinct populations</caption><thead><tr><th>Skill</th><th>Training tech</th><th>Muse {capped?'capped':'uncapped'}</th></tr></thead><tbody>{comparison.map(row=><tr key={row.skill}><th>{row.skill}</th><td>{formatPercent(row.train_tech_share)}</td><td>{formatPercent(capped?row.muse_share_capped:row.muse_share)}</td></tr>)}</tbody></table>}
        {source === 'adzuna' && a ? <>
          <p>{formatNumber(a.salary_label_status.estimated_rows)} cached targets use estimated salaries; {formatNumber(a.salary_label_status.advertised_rows)} are advertised. This is cross-source diagnostic evidence, not independent salary verification.</p>
          <p>Conditional employer-bootstrap intervals: accuracy {formatScore(a.bootstrap.accuracy.low)}–{formatScore(a.bootstrap.accuracy.high)}; macro F1 {formatScore(a.bootstrap.macro_f1.low)}–{formatScore(a.bootstrap.macro_f1.high)}. Predictions remain fixed, so these intervals omit model-training and selection uncertainty.</p>
          <table><thead><tr><th scope="col">Comparator</th><th scope="col">Accuracy</th><th scope="col">Macro F1</th></tr></thead><tbody>
            <tr><th scope="row">Historical training-majority ({a.comparators.historical_training_majority.label})</th><td>{formatScore(a.comparators.historical_training_majority.accuracy)}</td><td>{formatScore(a.comparators.historical_training_majority.macro_f1)}</td></tr>
            <tr><th scope="row">Observed-majority oracle ({a.comparators.observed_majority_oracle.label})</th><td>{formatScore(a.comparators.observed_majority_oracle.accuracy)}</td><td>{formatScore(a.comparators.observed_majority_oracle.macro_f1)}</td></tr>
          </tbody></table><p className="subtle">The oracle uses observed evaluation labels and is not a training-only baseline. {a.roc_auc.reason}</p>
        </> : contract && m ? <>
          <p>Primary support ≥ {formatPercent(contract.primary_support_inclusive)}, confidence &gt; {formatPercent(contract.confidence_strictly_greater_than)}, lift &gt; {formatScore(contract.lift_strictly_greater_than)}. At least two extracted skills are required. The 25-per-employer cap is applied before eligibility; 2% support is a separate sensitivity check.</p>
          <p>Confirmed in this selection: {m.confirmed_primary_rules.join('; ') || 'none'}. Intervals resample employers with the rules held fixed. Capped and uncapped populations can confirm different rules.</p>
          {!!fixedRules.length && <table><thead><tr><th scope="col">Fixed rule</th><th scope="col">Support</th><th scope="col">Confidence</th><th scope="col">Lift</th><th scope="col">Primary</th></tr></thead><tbody>{fixedRules.map(rule => <tr key={rule.rule}><th scope="row">{rule.rule}</th><td>{formatPercent(rule.support)}</td><td>{formatPercent(rule.confidence)}</td><td>{formatScore(rule.lift)}</td><td>{ruleStatus(rule)}</td></tr>)}</tbody></table>}
        </> : null}
      </details>
    </>}
  </div>;
}
