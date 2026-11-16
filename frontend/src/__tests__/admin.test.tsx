/** Admin calculations are based on saved evidence, never model re-evaluation. */
import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import Live from '../pages/Live';
import { AdminOverview, AssociationMining, DataQuality } from '../pages/Admin';
import { emptyProductFixture, productFixture } from './fixtures';
vi.mock('../components/Chart', async importOriginal => {
  const actual = await importOriginal<typeof import('../components/Chart')>();
  return {...actual,default:({option,ariaLabel}:ComponentProps<typeof actual.default>)=><div role="img" aria-label={ariaLabel}><output>{JSON.stringify(option)}</output></div>};
});
function option(name:string) {
  return JSON.parse(within(screen.getByRole('img',{name})).getByRole('status').textContent ?? '{}') as {series:{name?:string,data:({value:number[]}|{value:number|null})[]}[]};
}
describe('Admin evidence views',()=>{
  it('keeps grouped and cached results in separately labeled evidence panels',()=>{
    render(<AdminOverview data={productFixture()}/>);
    expect(screen.getByRole('heading',{name:'Grouped evaluation'})).toBeInTheDocument();
    expect(screen.getByRole('heading',{name:'Cached cross-source validation'})).toBeInTheDocument();
    expect(screen.getByText('Presentation mode · no access control')).toBeInTheDocument();
  });
  it('states when saved mining payloads are missing without inventing rules',()=>{
    const data=emptyProductFixture();data.rules.general=null;
    render(<AssociationMining data={data}/>);
    expect(screen.getByRole('heading',{name:'Saved mining artifacts unavailable'})).toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });
  it('selects and ranks saved rules with accessible controls',async()=>{
    const data=productFixture();data.rules.general=[{antecedents:'python',consequents:'sql',support:.2,confidence:.8,lift:2}];
    const user=userEvent.setup();render(<AssociationMining data={data}/>);
    await user.selectOptions(screen.getByRole('combobox',{name:'Inspect association rule'}),'python → sql');
    expect(screen.getByRole('heading',{name:'python → sql'})).toBeInTheDocument();
    expect(screen.getAllByRole('article').find(item=>within(item).queryByText('Confidence'))).toHaveTextContent('80.0%');
  });
  it('derives quality coverage from compact rows and discloses duplicate-check limitations',()=>{
    const data=productFixture();data.rows[0][3]=[];data.locations[0]=null;
    render(<DataQuality data={data}/>);
    expect(screen.getByText('83.3%')).toBeInTheDocument();
    expect(screen.getByText(/Independent duplicate-job detection cannot be repeated here/)).toBeInTheDocument();
  });
  it('derives precision from saved confusion columns and switches matrix units',async()=>{
    const data=productFixture();
    data.live.evaluation!.adzuna.confusion_rows_actual_columns_predicted=[[2,1,0],[1,1,0],[0,0,1]];
    const user=userEvent.setup();render(<Live data={data}/>);
    const series=option('Cached Adzuna recall and F1 by salary tier').series.find(row=>row.name==='Precision · derived');
    expect(series?.data[0].value).toBeCloseTo(2/3);
    let matrix=option('Cached Adzuna confusion matrix');
    expect((matrix.series[0].data[0].value as number[])[2]).toBeCloseTo(2/3);
    await user.click(within(screen.getByRole('group',{name:'Cached confusion matrix units'})).getByRole('button',{name:'Counts'}));
    matrix=option('Cached Adzuna confusion matrix');
    expect((matrix.series[0].data[0].value as number[])[2]).toBe(2);
    await user.selectOptions(screen.getByRole('combobox',{name:'Inspect confusion cell'}),'0-1');
    expect(screen.getByText(/1 jobs observed as Low and predicted as Mid/)).toBeInTheDocument();
  });
  it('compares capped and uncapped rule populations without conflating them',async()=>{
    const user=userEvent.setup();render(<Live data={productFixture()}/>);
    await user.click(screen.getByRole('button',{name:'Muse'}));
    expect(screen.getByRole('heading',{name:'Employer concentration sensitivity'})).toBeInTheDocument();
    expect(screen.getByText('At most 25 per company')).toBeInTheDocument();
  });
});
