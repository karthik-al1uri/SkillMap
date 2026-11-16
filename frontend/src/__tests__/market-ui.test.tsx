/** Synthetic test postings only. Production UI is supplied by verified cached exports. */
import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import type { MarketData, MarketPosting } from '../data/market';
import LiveMarket from '../pages/LiveMarket';
import { productFixture } from './fixtures';
vi.mock('../components/Chart', async importOriginal => {
  const actual = await importOriginal<typeof import('../components/Chart')>();
  return { ...actual, default: ({option,ariaLabel,onClick}:ComponentProps<typeof actual.default>) => {
    const plain=JSON.parse(JSON.stringify(option)) as {series?:{data?:{skill?:string}[]}[]};
    return <div role="img" aria-label={ariaLabel}><output>{JSON.stringify(option)}</output>{onClick && plain.series?.[0]?.data?.map(item=>item.skill && <button key={item.skill} onClick={()=>onClick({data:item})}>Plot focus {item.skill}</button>)}</div>;
  }};
});
function fixture():MarketData {
  const posting = (id:string,title:string,skills:string[],salary:number|null,salaryStatus:MarketPosting['salaryStatus'],location='Test Boulder'):MarketPosting => ({id,title,company:'Synthetic company',location,date:'2026-01-01',skills,salary,salaryStatus,tier:salary===null?null:salary>100000?'High':'Mid'});
  return {
    schema:'skillmap.market.v1',
    sources:{
      adzuna:{label:'Adzuna',postings:[posting('a','Synthetic Engineer',['python','sql'],120000,'estimated'),posting('b','Synthetic Analyst',['sql'],80000,'advertised','Test Denver'),posting('c','Synthetic Generalist',['healthcare'],null,'unavailable')],provenance:{path:'synthetic-adzuna.csv',bytes:0,sha256:'0'.repeat(64)},limitations:['Synthetic test fixture']},
      muse:{label:'The Muse',postings:[posting('m','Synthetic Platform Role',['python','security'],null,'unavailable')],provenance:{path:'synthetic-muse.csv',bytes:0,sha256:'0'.repeat(64)},limitations:['Synthetic test fixture']},
    },provenance:{approved_html:{path:'synthetic.html',bytes:0,sha256:'0'.repeat(64)},source_checks_requested:true,execution:'Synthetic test fixture',posting_grain:'(source, id, collection date)',salary_currency:'USD',salary_period:'annual',refresh_status:'No refresh authorized'},
  };
}
function option(label:string) { const chart=screen.getByRole('img',{name:label});return JSON.parse(within(chart).getByRole('status').textContent ?? '{}'); }

describe('user-facing Live Market',()=>{
  it('shows saved collection dates, explicit stale state and observed salary coverage',()=>{
    render(<LiveMarket data={productFixture()} market={fixture()}/>);
    expect(screen.getByRole('heading',{name:'The latest available, in focus.'})).toBeInTheDocument();
    expect(within(screen.getByRole('region',{name:'Market snapshot summary'})).getByText('66.7%')).toBeInTheDocument();
    expect(screen.getByText(/1 advertised · 1 estimated/)).toBeInTheDocument();
    expect(screen.getByText(/Older than the 30-day freshness threshold/)).toBeInTheDocument();
    expect(screen.queryByText('ROC AUC')).not.toBeInTheDocument();
  });
  it('uses chart focus to cross-filter saved postings and can reset',async()=>{
    const user=userEvent.setup();render(<LiveMarket data={productFixture()} market={fixture()}/>);
    await user.click(screen.getByRole('button',{name:'Plot focus python'}));
    expect(screen.getByText('1 matching records')).toBeInTheDocument();
    expect(screen.getByRole('heading',{name:'Synthetic Engineer'})).toBeInTheDocument();
    expect(screen.queryByRole('heading',{name:'Synthetic Analyst'})).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:/sql.*shared postings/})).toHaveTextContent('50.0% Jaccard');
    await user.click(screen.getByRole('button',{name:'Clear skill focus'}));
    expect(screen.getByText('3 matching records')).toBeInTheDocument();
  });
  it('switches sources without carrying focus and never invents Muse salaries',async()=>{
    const user=userEvent.setup();render(<LiveMarket data={productFixture()} market={fixture()}/>);
    await user.click(screen.getByRole('button',{name:'Plot focus sql'}));
    await user.click(within(screen.getByRole('group',{name:'Market source'})).getByRole('button',{name:'The Muse'}));
    expect(screen.getByRole('heading',{name:'Salary was not supplied by The Muse'})).toBeInTheDocument();
    expect(screen.getByRole('heading',{name:'Synthetic Platform Role'})).toBeInTheDocument();
    expect(screen.queryByRole('img',{name:'Cached posting salary distribution'})).not.toBeInTheDocument();
  });
  it('filters salary evidence and locations using actual saved rows',async()=>{
    const user=userEvent.setup();render(<LiveMarket data={productFixture()} market={fixture()}/>);
    await user.selectOptions(screen.getByRole('combobox',{name:'Market salary evidence'}),'advertised');
    expect(screen.getByText('$80,000')).toBeInTheDocument();
    const histogram=option('Cached posting salary distribution') as {series:{data:number[]}[]};
    expect(histogram.series[0].data.reduce((a,b)=>a+b,0)).toBe(1);
  });
  it('searches postings and opens an accessible saved-record drawer',async()=>{
    const user=userEvent.setup();render(<LiveMarket data={productFixture()} market={fixture()}/>);
    await user.type(screen.getByRole('searchbox',{name:'Search saved postings'}),'Synthetic Analyst');
    expect(screen.getByText('1 matching records')).toBeInTheDocument();
    await user.click(screen.getByRole('button',{name:'Inspect saved posting →'}));
    const dialog=screen.getByRole('dialog',{name:'Synthetic Analyst'});
    expect(within(dialog).getByText('Advertised salary in saved source')).toBeInTheDocument();
    expect(within(dialog).getByRole('button',{name:'Close details'})).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:'Inspect saved posting →'})).toHaveFocus();
  });
  it('provides technical/general classification and searchable skill selection',async()=>{
    const user=userEvent.setup();render(<LiveMarket data={productFixture()} market={fixture()}/>);
    await user.selectOptions(screen.getByRole('combobox',{name:'Skill classification'}),'technical');
    const ranking=option('Cached posting skill ranking') as {yAxis:{data:string[]}};
    expect(ranking.yAxis.data).toEqual(['python','sql']);
    const search=screen.getByRole('combobox',{name:'Explore a market skill'});
    await user.click(search);await user.type(search,'python');await user.keyboard('{Enter}');
    expect(screen.getByText('1 matching records')).toBeInTheDocument();
  });
  it('paginates saved cards without inventing or duplicating records',async()=>{
    const market=fixture();
    market.sources.adzuna.postings=Array.from({length:8},(_,index)=>({...market.sources.adzuna.postings[0],id:`synthetic-${index}`,title:`Synthetic Role ${index}`}));
    const user=userEvent.setup();render(<LiveMarket data={productFixture()} market={market}/>);
    expect(screen.getAllByRole('button',{name:'Inspect saved posting →'})).toHaveLength(6);
    await user.click(screen.getByRole('button',{name:'Next'}));
    expect(screen.getAllByRole('button',{name:'Inspect saved posting →'})).toHaveLength(2);
    expect(screen.getByText('Page 2 of 2')).toBeInTheDocument();
  });
  it('shows honest unavailable and no-results states',async()=>{
    const user=userEvent.setup();const {rerender}=render(<LiveMarket data={productFixture()} market={null} marketError="Saved export unavailable"/>);
    expect(screen.getByRole('heading',{name:'Market snapshot unavailable'})).toBeInTheDocument();
    rerender(<LiveMarket data={productFixture()} market={fixture()}/>);
    await user.type(screen.getByRole('searchbox',{name:'Search saved postings'}),'no-such-posting');
    expect(screen.getByRole('heading',{name:'No saved postings match'})).toBeInTheDocument();
  });
});
