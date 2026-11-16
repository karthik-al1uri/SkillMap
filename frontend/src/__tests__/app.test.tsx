/** React interactions use a mocked chart; real chart configurations are tested separately. */
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import App from '../App';
import { emptyProductFixture, productFixture } from './fixtures';
import { marketFixture } from './market-fixture';
import type { ProductData } from '../data/types';
import { navigate } from '../routes';

vi.mock('../components/Chart', async importOriginal=>{
  const actual=await importOriginal<typeof import('../components/Chart')>();
  return {...actual,default:({option,ariaLabel,onClick}:ComponentProps<typeof actual.default>)=>{
    const plain=JSON.parse(JSON.stringify(option)) as {series?:{type?:string;data?:{skillId?:number;name?:string}[]}[]};
    const nodes=plain.series?.find(series=>series.type==='graph')?.data??[];
    return <div role="img" aria-label={ariaLabel}><output data-testid="chart-option">{JSON.stringify(option)}</output>{onClick&&nodes.map(node=><button key={node.skillId} onClick={()=>onClick({data:node})}>Mock select {node.name}</button>)}</div>;
  }};
});
function successfulFetch(data:ProductData=productFixture()) {
  return vi.fn().mockImplementation((url:string)=>Promise.resolve({ok:true,status:200,json:async()=>url.endsWith('market.json')?marketFixture():data}));
}
async function loadPage(path:string,data:ProductData=productFixture()) {
  window.history.replaceState(null,'',path);const fetch=successfulFetch(data);vi.stubGlobal('fetch',fetch);render(<App/>);
  await screen.findByRole('heading',{level:1});return {user:userEvent.setup(),fetch};
}
function stat(label:string){return screen.getAllByRole('article').find(item=>within(item).queryByText(label,{exact:true}))!;}
function chartOption(label:string){return JSON.parse(within(screen.getByRole('img',{name:label})).getByTestId('chart-option').textContent??'{}');}
async function chooseSearch(user:ReturnType<typeof userEvent.setup>,label:string,option:string){const input=screen.getByRole('combobox',{name:label});await user.click(input);await user.type(input,option);await user.click(screen.getByRole('option',{name:option}));}
beforeEach(()=>window.history.replaceState(null,'','/app'));
afterEach(()=>{vi.unstubAllGlobals();vi.useRealTimers();});

describe('two application modes',()=>{
  it('navigates all User and Admin routes without refetching data',async()=>{
    const {user,fetch}=await loadPage('/app');
    for(const name of ['Explore','Skills','Salary Intelligence','Career Profiles','Live Market']){
      await user.click(within(screen.getByRole('navigation')).getByRole('link',{name: new RegExp(`^${name}`)}));
      await screen.findByRole('heading',{level:1});
      expect(screen.getByRole('link',{name:new RegExp(`^${name}`),current:'page'})).toBeInTheDocument();
      expect(location.pathname.startsWith('/app')).toBe(true);
    }
    await user.click(screen.getByRole('button',{name:'Admin'}));
    for(const name of ['Overview','Model Evaluation','Association Mining','Clustering Analysis','Data Quality','Live Validation']){
      await user.click(within(screen.getByRole('navigation')).getByRole('link',{name}));
      expect(screen.getByRole('link',{name,current:'page'})).toBeInTheDocument();
      expect(location.pathname.startsWith('/admin')).toBe(true);
    }
    expect(fetch.mock.calls.filter(([url])=>String(url).endsWith('skillmap.json'))).toHaveLength(1);
    expect(fetch.mock.calls.filter(([url])=>String(url).endsWith('market.json'))).toHaveLength(1);
    expect(screen.getByText('Research presentation · not an access-control boundary')).toBeInTheDocument();
  });
  it('keeps technical model and clustering diagnostics out of User mode',async()=>{
    const {user}=await loadPage('/app/profiles');
    expect(screen.queryByText(/silhouette|seed ARI|ROC AUC|macro F1/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('link',{name:'Model Evaluation'})).not.toBeInTheDocument();
    await user.click(screen.getByRole('link',{name:'Skills'}));
    expect(screen.queryByText('Inspect the strongest saved association rules')).not.toBeInTheDocument();
    await user.click(screen.getByRole('button',{name:'Admin'}));
    await user.click(screen.getByRole('link',{name:'Clustering Analysis'}));
    expect(screen.getByRole('heading',{name:'How distinct are the groups?'})).toBeInTheDocument();
  });
  it('supports direct Admin links and browser history events',async()=>{
    const {user}=await loadPage('/admin/model');
    expect(await screen.findByText('Grouped macro F1')).toBeInTheDocument();
    await user.click(screen.getByRole('button',{name:'User'}));
    expect(location.pathname).toBe('/app');
    window.history.replaceState(null,'','/admin/quality');window.dispatchEvent(new PopStateEvent('popstate'));
    expect(await screen.findByRole('heading',{name:'Know what the evidence can support.'})).toBeInTheDocument();
  });
  it('retains legacy links and exposes a real not-found state',async()=>{
    await loadPage('/#model');expect(location.pathname).toBe('/admin/model');
    navigate('/app/not-a-route');expect(await screen.findByRole('heading',{name:'Page not found'})).toBeInTheDocument();
  });
});
describe('loading and missing data',()=>{
  it('shows pending state',()=>{vi.stubGlobal('fetch',vi.fn(()=>new Promise(()=>{})));render(<App/>);expect(screen.getByRole('status',{name:'Loading saved evidence'})).toBeInTheDocument();});
  it('handles historical dataset failure and retries without fabricating data',async()=>{
    let failed=true;vi.stubGlobal('fetch',vi.fn().mockImplementation((url:string)=>Promise.resolve(url.endsWith('market.json')?{ok:true,json:async()=>marketFixture()}:failed?{ok:false,status:404}:{ok:true,json:async()=>productFixture()})));
    render(<App/>);expect(await screen.findByRole('alert')).toHaveTextContent('HTTP 404');failed=false;await userEvent.click(screen.getByRole('button',{name:'Retry data loading'}));expect(await screen.findByRole('heading',{name:'Find the signal in the skills.'})).toBeInTheDocument();
  });
  it('isolates missing market data from historical exploration',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockImplementation((url:string)=>Promise.resolve(url.endsWith('market.json')?{ok:false,status:404}:{ok:true,json:async()=>productFixture()})));
    window.history.replaceState(null,'','/app/live');render(<App/>);expect(await screen.findByRole('heading',{name:'Market snapshot unavailable'})).toBeInTheDocument();
    await userEvent.click(screen.getByRole('link',{name:'Explore'}));expect(await screen.findByRole('heading',{name:'Find the signal in the skills.'})).toBeInTheDocument();
  });
  it('rejects corrupt nested model evidence before rendering metrics',async()=>{const data=productFixture();data.models.grouped!.report.outer_metrics.confusion_matrix=[[-1]];vi.stubGlobal('fetch',successfulFetch(data));render(<App/>);expect(await screen.findByRole('alert')).toHaveTextContent('3x3 nonnegative integer counts');});
  it('shows explicit unavailable grouped results',async()=>{await loadPage('/admin/model',emptyProductFixture());expect(screen.getByRole('heading',{name:'Grouped evaluation unavailable'})).toBeInTheDocument();});
});
describe('user exploration interactions',()=>{
  it('cross-filters Overview and clears family focus',async()=>{const {user}=await loadPage('/app');const maximum=()=>Math.max(...chartOption('Most frequently mentioned skills').series[0].data);expect(maximum()).toBe(4);await user.click(screen.getByRole('button',{name:/Synthetic Technical Ops/}));expect(maximum()).toBe(3);await user.click(screen.getByRole('button',{name:'Clear focus'}));expect(maximum()).toBe(4);});
  it('searches the network, uses separate overlap measures and resets',async()=>{const {user}=await loadPage('/app/skills');await user.click(screen.getByRole('button',{name:'Mock select healthcare'}));expect(screen.getByRole('combobox',{name:'Inspect a skill'})).toHaveValue('healthcare');await user.click(screen.getByRole('button',{name:'Reset network'}));await user.click(screen.getByRole('button',{name:'Jaccard'}));expect(chartOption("Shared-posting counts for the selected skill's neighbors").series[0].data.at(-1)).toBeCloseTo(.4);await chooseSearch(user,'Inspect a skill','sql');expect(screen.getByRole('heading',{name:'Alongside sql'})).toBeInTheDocument();await user.click(screen.getByRole('button',{name:'Technical'}));expect(chartOption('Observed skill co-occurrence network').series[0].data).toHaveLength(2);});
  it('combines salary filters at unique-job grain, handles empty selection and resets',async()=>{const {user}=await loadPage('/app/salary');await chooseSearch(user,'Archetype','Synthetic Technical Ops');expect(stat('Median annual salary')).toHaveTextContent('$80,000');await chooseSearch(user,'Skill mentioned','healthcare');expect(stat('Jobs in your selection')).toHaveTextContent('0');expect(stat('Median annual salary')).toHaveTextContent('—');expect(screen.getByRole('heading',{name:'No matching jobs'})).toBeInTheDocument();await user.click(screen.getByRole('button',{name:'Reset filters'}));expect(stat('Jobs in your selection')).toHaveTextContent('6');await user.click(screen.getByRole('button',{name:'Full range'}));expect(chartOption('Salary distribution of the filtered unique-job cohort').series[0].data.reduce((a:number,b:number)=>a+b,0)).toBe(6);});
  it('selects a career profile without exposing diagnostics',async()=>{const {user}=await loadPage('/app/profiles');await user.selectOptions(screen.getByRole('combobox',{name:'Explore an archetype'}),'1');expect(stat('Jobs in this archetype')).toHaveTextContent('Synthetic Care Services');expect(screen.queryByRole('heading',{name:'How distinct are the groups?'})).not.toBeInTheDocument();});
  it('adapts network layout to compact media queries',async()=>{vi.stubGlobal('matchMedia',vi.fn(()=>({matches:true,addEventListener:vi.fn(),removeEventListener:vi.fn()})));await loadPage('/app/skills');const node=chartOption('Observed skill co-occurrence network').series[0].data[3];expect(node.x).toBe(0);expect(node.y).toBeGreaterThan(0);});
  it('supports mobile focus containment, Escape and arrow navigation',async()=>{const {user}=await loadPage('/app');await user.click(screen.getByRole('button',{name:'Open menu'}));expect(screen.getByRole('link',{name:'Explore'})).toHaveFocus();await user.keyboard('{Tab}');expect(screen.getByRole('link',{name:'Skills'})).toHaveFocus();await user.keyboard('{Escape}');expect(screen.getByRole('button',{name:'Open menu'})).toHaveFocus();screen.getByRole('link',{name:'Explore'}).focus();await user.keyboard('{ArrowDown}');expect(screen.getByRole('link',{name:'Skills'})).toHaveFocus();});
});
