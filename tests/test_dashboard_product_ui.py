"""Independent product UI checks using static HTML and Node VM, never a real browser."""
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(os.environ.get("SKILLMAP_ASSET_ROOT", Path(__file__).resolve().parents[1]))
ASSETS = ROOT / "src/dashboard_assets"
ROWS = [[100,0,0,[0],0],[200,0,1,[1],1],[300,1,0,[0,1],2],[400,1,1,[],2]]


@pytest.fixture(scope="module")
def node():
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Existing Node executable unavailable; no dependency installation")
    return executable


def run_js(node, expression, setup=""):
    source = ASSETS / "product.js"
    assert source.is_file(), "Product JavaScript must exist before UI verification"
    runner = r'''
const fs = require('node:fs');
const vm = require('node:vm');
const sandbox = {console: {log(){}, warn(){}, error(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox, {timeout: 5000});
const SkillMap = sandbox.SkillMap || (sandbox.window && sandbox.window.SkillMap);
if (!SkillMap) throw new Error('SkillMap helpers were not exposed without a DOM');
''' + setup + '\nconst result = (' + expression + ');\nprocess.stdout.write(JSON.stringify(result));'
    result = subprocess.run([node, "-e", runner, str(source)], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("values,expected", [([],None),([9],9),([3,1,2],2),([4,1,2,3],2.5)])
def test_median_boundaries_and_source_preservation(node, values, expected):
    result=run_js(node,"({median:SkillMap.median(values),values})",f"const values={json.dumps(values)};")
    assert result["median"] == expected
    assert result["values"] == values


def test_salary_cohort_combines_filters_and_handles_zero_indices(node):
    result=run_js(node, "({all:SkillMap.salaryCohort(rows,{role:null,location:null,skill:null}),role:SkillMap.salaryCohort(rows,{role:0,location:null,skill:null}),location:SkillMap.salaryCohort(rows,{role:null,location:0,skill:null}),skill:SkillMap.salaryCohort(rows,{role:null,location:null,skill:0}),combined:SkillMap.salaryCohort(rows,{role:0,location:0,skill:0}),empty:SkillMap.salaryCohort(rows,{role:0,location:0,skill:1}),rows})",f"const rows={json.dumps(ROWS)};")
    assert result["all"] == ROWS
    assert result["role"] == ROWS[:2]
    assert result["location"] == [ROWS[0],ROWS[2]]
    assert result["skill"] == [ROWS[0],ROWS[2]]
    assert result["combined"] == ROWS[:1]
    assert result["empty"] == []
    assert result["rows"] == ROWS


@pytest.mark.parametrize("dimension,expected",[
    ("role",[{"key":1,"count":2,"median":350},{"key":0,"count":2,"median":150}]),
    ("location",[{"key":1,"count":2,"median":300},{"key":0,"count":2,"median":200}]),
    ("skill",[{"key":1,"count":2,"median":250},{"key":0,"count":2,"median":200}]),
])
def test_salary_group_denominators_and_postcohort_minimum(node,dimension,expected):
    result=run_js(node,f"({{groups:SkillMap.salaryGroups(rows,{json.dumps(dimension)},2),suppressed:SkillMap.salaryGroups(rows,{json.dumps(dimension)},3)}})",f"const rows={json.dumps(ROWS)};")
    assert result["groups"] == expected
    assert result["suppressed"] == []


def test_salary_group_ties_are_deterministic_and_missing_cohort_is_not_zero(node):
    result=run_js(node,"({groups:SkillMap.salaryGroups(rows,'role',1),empty:SkillMap.median(SkillMap.salaryCohort(rows,{role:99,location:null,skill:null}).map(row=>row[0]))})","const rows=[[100,1,0,[],0],[100,0,0,[],0]];")
    assert result["groups"] == [{"key":0,"count":1,"median":100},{"key":1,"count":1,"median":100}]
    assert result["empty"] is None


def test_confusion_count_mode_does_not_mutate_source(node):
    matrix=[[1,1,0],[0,0,0],[0,1,3]]
    result=run_js(node,"({counts:SkillMap.confusion(matrix,false),matrix})",f"const matrix={json.dumps(matrix)};")
    assert result["counts"] == matrix
    assert result["matrix"] == matrix


def test_network_selection_returns_only_neighbors_sorted_by_count(node):
    graph={"nodes":[{"id":0,"name":"zero"},{"id":1,"name":"one"},{"id":2,"name":"two"},{"id":3,"name":"isolated"}],"edges":[{"source":0,"target":1,"count":2,"jaccard":.2},{"source":2,"target":0,"count":5,"jaccard":.5},{"source":1,"target":2,"count":9,"jaccard":.9}]}
    result=run_js(node,"({selected:SkillMap.networkSelection(graph,0),isolated:SkillMap.networkSelection(graph,3),unknown:SkillMap.networkSelection(graph,99)})",f"const graph={json.dumps(graph)};")
    assert result["selected"]["node"] == graph["nodes"][0]
    assert result["selected"]["neighbors"] == [{"id":2,"count":5,"jaccard":.5},{"id":1,"count":2,"jaccard":.2}]
    assert result["isolated"]["node"] == graph["nodes"][3]
    assert result["isolated"]["neighbors"] == []
    assert result["unknown"] == {"node":None,"neighbors":[]}


def test_confusion_row_share_normalizes_each_observed_tier_and_zero_rows(node):
    matrix=[[1,1,0],[0,0,0],[0,1,3]]
    result=run_js(node,"({share:SkillMap.confusion(matrix,true),countsAreSource:SkillMap.confusion(matrix,false)===matrix,matrix})",f"const matrix={json.dumps(matrix)};")
    assert result["share"] == [[.5,.5,0],[0,0,0],[0,.25,.75]]
    assert result["countsAreSource"] is False
    assert result["matrix"] == matrix


class ProductTemplateInspector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements=[]
        self.stack=[]
        self.text=[]
        self.primary={key:[] for key in ("overview","skills","salary","clusters","live","model")}
        self.secondary=[]
        self.panel_ids=[]
        self.paragraphs=[]
        self.paragraph=None

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        self.elements.append((tag,attrs))
        if attrs.get("role")=="tabpanel":
            self.panel_ids.append(attrs.get("id"))
        panel=next((a.get("id") for t,a in reversed(self.stack) if a.get("role")=="tabpanel"),None)
        if attrs.get("data-primary-chart")=="true":
            self.primary.setdefault(panel,[]).append(attrs.get("id"))
        if attrs.get("data-secondary-chart")=="true":
            self.secondary.append((attrs.get("id"),any(t in ("details","dialog") for t,a in self.stack)))
        if tag=="p":
            self.paragraph={"text":[],"secondary":any(t in ("details","dialog") for t,a in self.stack)}
            self.paragraphs.append(self.paragraph)
        if tag not in ("area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"):
            self.stack.append((tag,attrs))

    def handle_endtag(self,tag):
        if tag=="p":
            self.paragraph=None
        for i in range(len(self.stack)-1,-1,-1):
            if self.stack[i][0]==tag:
                del self.stack[i:]
                break

    def handle_data(self,text):
        if self.paragraph is not None:
            self.paragraph["text"].append(text)


def template_inspection():
    template=(ASSETS / "product.html").read_text()
    inspector=ProductTemplateInspector()
    inspector.feed(template)
    return template,inspector


def test_template_six_views_have_focused_primary_charts_and_collapsed_secondary_evidence():
    template,inspector=template_inspection()
    assert inspector.panel_ids==["overview","skills","salary","clusters","live","model"]
    for panel,charts in inspector.primary.items():
        assert 1 <= len(charts) <= 3,(panel,charts)
        assert len(charts)==len(set(charts))
    assert inspector.secondary
    assert all(inside_details for _,inside_details in inspector.secondary)
    for paragraph in inspector.paragraphs:
        if not paragraph["secondary"]:
            assert len(" ".join("".join(paragraph["text"]).split())) <= 240
    dialogs=[attrs for tag,attrs in inspector.elements if tag=="dialog"]
    assert len(dialogs)==1 and dialogs[0].get("aria-labelledby")


def test_template_filter_controls_have_visible_or_aria_labels_and_no_remote_assets():
    template,inspector=template_inspection()
    labels={attrs["for"] for tag,attrs in inspector.elements if tag=="label" and attrs.get("for")}
    controls={attrs.get("id"):attrs for tag,attrs in inspector.elements if tag in ("input","select")}
    assert {"salary-role","salary-location","salary-skill","salary-min","network-search","cluster-select","live-cap"} <= set(controls)
    assert all(identifier in labels or attrs.get("aria-label") or attrs.get("aria-labelledby") for identifier,attrs in controls.items())
    for tag,attrs in inspector.elements:
        if tag=="script":
            assert not attrs.get("src")
        if tag=="link":
            assert not attrs.get("href") or attrs["href"].startswith("data:")
        if attrs.get("src"):
            assert attrs["src"].startswith("data:")
    for key in ("network-scope","salary-dimension","live-source","confusion-mode"):
        buttons=[attrs for tag,attrs in inspector.elements if tag=="button" and f"data-{key}" in attrs]
        assert len(buttons)>=2
        assert all(attrs.get("aria-pressed") in ("true","false") for attrs in buttons)
        assert sum(attrs["aria-pressed"]=="true" for attrs in buttons)==1


def controller_payload():
    graph={"nodes":[{"id":0,"label":"python","count":2,"x":-.5,"y":0,"role":0},{"id":1,"label":"sql","count":2,"x":.5,"y":0,"role":1}],"edges":[{"source":0,"target":1,"count":1,"jaccard":1/3}]}
    report={"winner":"nominal_logistic","cv_rows":[{"candidate":"nominal_logistic","mean_f1_macro":.7},{"candidate":"fixed_xgboost","mean_f1_macro":.6}],"outer_metrics":{"f1_macro":.4,"accuracy":.5,"roc_auc_ovr_macro":.6,"f1_per_class":{"Low":.5,"Mid":.4,"High":.8},"confusion_matrix":[[1,1,0],[0,0,0],[0,1,3]]}}
    adzuna={"n":4,"accuracy":.5,"macro_f1":.4,"actual_counts":{"Low":1,"Mid":1,"High":2},"predicted_counts":{"Low":2,"Mid":0,"High":2},"per_class_recall":{"Low":.5,"Mid":0,"High":1},"per_class_f1":{"Low":.4,"Mid":0,"High":.8},"salary_label_status":{"estimated_rows":3,"advertised_rows":1},"bootstrap":{"accuracy":{"low":.3,"high":.7},"macro_f1":{"low":.2,"high":.6}},"comparators":{"observed_majority_oracle":{"label":"High","accuracy":.5,"macro_f1":.3},"historical_training_majority":{"label":"Low","accuracy":.25,"macro_f1":.2}}}
    rule={"rule":"python → sql","support":.5,"lift":2,"confidence":.75,"passes_primary_005":True,"support_ci_low":.2,"support_ci_high":.7,"bootstrap_pass_fraction_primary_005":.8}
    capped_rule={**rule,"support":.25,"lift":1.1,"passes_primary_005":False}
    muse={"eligible_postings":3,"all_postings":4,"primary_confirmed":1,"fixed_rules":1,"employers_eligible":2,"sensitivity_confirmed":1,"confirmed_primary_rules":["python → sql"]}
    live={"adzuna":adzuna,"muse_uncapped":muse,"muse_company_capped":{**muse,"eligible_postings":2,"all_postings":3,"primary_confirmed":0,"confirmed_primary_rules":[]},"predeclared_contract":{"primary_support_inclusive":.05,"sensitivity_support_inclusive":.02,"confidence_strictly_greater_than":.5,"lift_strictly_greater_than":1.5,"muse_eligibility_min_skills":2}}
    return {"metrics":{"n_jobs":4,"median_salary":250,"n_skills_vocab":2,"mean_skills":1,"zero_skills":1,"salary_tier_boundaries":{"mid_max":109200},"live_snapshot_dates":["2026-10-14"]},"rows":ROWS,"roles":[{"name":"Role zero","count":2,"median_salary":150,"high_share":0,"top_skills":[{"skill":"python","share":.5,"overall_share":.5}]},{"name":"Role one","count":2,"median_salary":350,"high_share":1,"top_skills":[{"skill":"sql","share":.5,"overall_share":.5}]}],"locations":["CO","CA"],"skills":[{"index":0,"name":"python"},{"index":1,"name":"sql"}],"tiers":["Low","Mid","High"],"networks":{"all":graph,"technical":graph},"skill_neighbors":{"0":[{"skillIndex":1,"count":1}],"1":[{"skillIndex":0,"count":1}]},"rules":{"general":[],"technical":[]},"models":{"grouped":{"report":report},"historical":[{"model":"Historical test winner","f1_macro":.99}]},"robustness":{"summary":[]},"live":{"evaluation":{"report":live},"comparison":[{"skill":"python","muse_share":.5,"muse_share_capped":.25,"train_tech_share":.2}],"fixed_rules_uncapped":[rule],"fixed_rules_capped":[capped_rule]}}


@pytest.fixture(scope="module")
def controlled_dom_result(node):
    """Exercise event wiring in a small controlled stub; this is not browser rendering."""
    payload=controller_payload()
    harness=r'''
const fs=require('node:fs'),vm=require('node:vm');
(async()=>{
const D=JSON.parse(process.argv[2]);
const elements=new Map(),calls=new Map(),groups={};
let focused=null,selectedHash='',networkCalls=0;
function element(id='',dataset={}){
  const el={id,dataset,hidden:false,open:false,value:'',textContent:'',innerHTML:'',attrs:{},listeners:{},plotListeners:{},
    classList:{add(){},remove(){}},setAttribute(name,value){this.attrs[name]=value;},
    getAttribute(name){return this.attrs[name];},addEventListener(name,fn){this.listeners[name]=fn;},
    insertAdjacentHTML(where,text){this.innerHTML+=text;},focus(){focused=this.id;},
    on(name,fn){this.plotListeners[name]=fn;},removeAllListeners(name){delete this.plotListeners[name];},
    showModal(){this.open=true;},close(){this.open=false;}};
  if(id)elements.set(id,el);return el;
}
const ids=[...fs.readFileSync(process.argv[3],'utf8').matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);ids.forEach(id=>element(id));
const tabs=['overview','skills','salary','clusters','live','model'].map(id=>element('tab-'+id,{tab:id}));
const panels=['overview','skills','salary','clusters','live','model'].map(id=>elements.get(id));
groups['nav button']=tabs;groups['.panel']=panels;
function segment(selector,key,values){groups[selector]=values.map(value=>element('',{[key]:value}));}
segment('[data-network-scope]','networkScope',['all','technical']);
segment('[data-salary-dimension]','salaryDimension',['skill','role','location']);
segment('[data-live-source]','liveSource',['adzuna','muse']);
segment('[data-confusion-mode]','confusionMode',['counts','share']);
segment('[data-open-evidence]','openEvidence',['one','two']);
segment('[data-close-dialog]','closeDialog',['close']);
elements.get('skillmap-product-data').textContent=JSON.stringify(D);
elements.get('skillmap-provenance').textContent=JSON.stringify({sources:{fixture:{path:'fixture.csv',status:'synthetic_fixture',sha256:'a'.repeat(64)}}});
elements.get('salary-min').value='25';elements.get('live-cap').value='all';
const document={getElementById(id){if(!elements.has(id))throw Error('Unexpected DOM ID '+id);return elements.get(id);},querySelectorAll(selector){
 if(groups[selector])return groups[selector];
 const match=selector.match(/^#(\w+) \.plotly-graph-div$/);if(match)return [];
 throw Error('Unexpected selector '+selector);
}};
const sandbox={document,console,localStorage:{getItem(){return null;},setItem(){}},location:{hash:''},history:{replaceState(a,b,hash){selectedHash=hash;}},requestAnimationFrame(fn){fn();},
 Plotly:{react(el,traces,layout,config){calls.set(el.id,{traces,layout,config});if(el.id==='skills-network')networkCalls++;return Promise.resolve(el);},purge(el){calls.delete(el.id);},Plots:{resize(){}}}};
vm.createContext(sandbox);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),sandbox,{timeout:5000});
await new Promise(resolve=>setImmediate(resolve));
const fire=(el,name,extra={})=>{if(!el.listeners[name])throw Error('Missing listener '+name+' on '+el.id);el.listeners[name]({target:el,currentTarget:el,preventDefault(){},...extra});};
const find=(selector,value)=>groups[selector].find(el=>Object.values(el.dataset).includes(value));
const result={initialView:sandbox.SkillMap.state.view,cachedBadge:elements.get('snapshot-badge').textContent,cachedDate:elements.get('live-date').textContent};
fire(tabs[0],'keydown',{key:'ArrowRight'});result.keyboard={view:sandbox.SkillMap.state.view,focused,hidden:panels.map(el=>el.hidden),aria:tabs.map(el=>el.attrs['aria-selected'])};
elements.get('network-search').value='sql';fire(elements.get('network-search'),'input');result.networkSearch={focus:sandbox.SkillMap.state.focus,selected:elements.get('selected-skill').textContent,feedback:elements.get('network-feedback').textContent};
fire(find('[data-network-scope]','technical'),'click');result.networkScope={scope:sandbox.SkillMap.state.network,search:elements.get('network-search').value,pressed:groups['[data-network-scope]'].map(el=>el.attrs['aria-pressed']),calls:networkCalls};
await new Promise(resolve=>setImmediate(resolve));
elements.get('skills-network').plotListeners.plotly_click({points:[{customdata:1}]});result.networkNodeClick={focus:sandbox.SkillMap.state.focus,selected:elements.get('selected-skill').textContent};
fire(tabs[2],'click');elements.get('salary-min').value='1';fire(elements.get('salary-min'),'change');
result.salaryAll={stats:elements.get('salary-stats').innerHTML,comparison:calls.get('salary-comparison').traces[0],distribution:calls.get('salary-distribution').traces[0].x};
elements.get('salary-role').value='0';fire(elements.get('salary-role'),'change');
result.salaryRole={stats:elements.get('salary-stats').innerHTML,distribution:calls.get('salary-distribution').traces[0].x};
elements.get('salary-location').value='0';fire(elements.get('salary-location'),'change');result.salaryCombined=calls.get('salary-distribution').traces[0].x;
elements.get('salary-skill').value='1';fire(elements.get('salary-skill'),'change');result.salaryEmpty={stats:elements.get('salary-stats').innerHTML,distribution:elements.get('salary-distribution').textContent,comparison:elements.get('salary-comparison').textContent,stillPlotted:calls.has('salary-distribution')};
fire(elements.get('salary-reset'),'click');result.salaryReset={role:elements.get('salary-role').value,location:elements.get('salary-location').value,skill:elements.get('salary-skill').value,min:elements.get('salary-min').value,distribution:calls.get('salary-distribution').traces[0].x};
fire(tabs[3],'click');elements.get('cluster-select').value='1';fire(elements.get('cluster-select'),'change');result.cluster={role:sandbox.SkillMap.state.role,name:elements.get('cluster-name').textContent};
fire(tabs[4],'click');result.liveAdzuna={title:elements.get('live-left-title').textContent,traces:calls.get('live-left').traces};
fire(find('[data-live-source]','muse'),'click');result.liveMuse={title:elements.get('live-left-title').textContent,capHidden:elements.get('live-cap-wrapper').hidden,share:calls.get('live-left').traces[1].x,stats:elements.get('live-stats').innerHTML};
elements.get('live-cap').value='capped';fire(elements.get('live-cap'),'change');result.liveCapped={share:calls.get('live-left').traces[1].x,stats:elements.get('live-stats').innerHTML};
await new Promise(resolve=>setImmediate(resolve));
const ruleClick=elements.get('live-right').plotListeners.plotly_click;ruleClick({points:[{customdata:D.live.fixed_rules_capped[0]}]});result.liveRuleDetail=elements.get('live-rule-detail').innerHTML;
fire(tabs[5],'click');result.model={stats:elements.get('model-stats').innerHTML,counts:calls.get('model-confusion').traces[0].z,caveat:elements.get('model-caveat').innerHTML};
fire(find('[data-confusion-mode]','share'),'click');result.model.share=calls.get('model-confusion').traces[0].z;
elements.get('model-details').open=true;fire(elements.get('model-details'),'toggle');result.model.details={cv:calls.get('model-cv').traces[0].x,history:calls.get('model-history').traces[0].x};
fire(groups['[data-open-evidence]'][0],'click');result.dialogOpened=elements.get('evidence-dialog').open;fire(groups['[data-close-dialog]'][0],'click');result.dialogClosed=!elements.get('evidence-dialog').open;
process.stdout.write(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result=subprocess.run([node,"-e",harness,str(ASSETS/"product.js"),json.dumps(payload),str(ASSETS/"product.html")],capture_output=True,text=True,timeout=15,check=False)
    assert result.returncode==0,result.stderr
    return json.loads(result.stdout)


def test_controlled_dom_keyboard_navigation_network_search_and_scoping(controlled_dom_result):
    result=controlled_dom_result
    assert result["initialView"]=="overview"
    assert result["keyboard"]=={"view":"skills","focused":"tab-skills","hidden":[True,False,True,True,True,True],"aria":["false","true","false","false","false","false"]}
    assert result["networkSearch"]=={"focus":1,"selected":"sql","feedback":"Selected sql"}
    assert result["networkScope"]["scope"]=="technical"
    assert result["networkScope"]["search"]==""
    assert result["networkScope"]["pressed"]==["false","true"]
    assert result["networkScope"]["calls"]>=3
    assert result["networkNodeClick"]=={"focus":1,"selected":"sql"}


def test_controlled_dom_salary_events_recompute_cohort_and_empty_state(controlled_dom_result):
    result=controlled_dom_result
    assert result["salaryAll"]["distribution"]==[100,200,300,400]
    assert result["salaryAll"]["comparison"]["x"]==[200,250]
    assert result["salaryAll"]["comparison"]["customdata"]==[2,2]
    assert "$250" in result["salaryAll"]["stats"]
    assert result["salaryRole"]["distribution"]==[100,200]
    assert "$150" in result["salaryRole"]["stats"]
    assert result["salaryCombined"]==[100]
    assert result["salaryEmpty"]["stillPlotted"] is False
    assert "No jobs match" in result["salaryEmpty"]["distribution"]
    assert "—" in result["salaryEmpty"]["stats"]
    assert result["salaryReset"]=={"role":"","location":"","skill":"","min":"25","distribution":[100,200,300,400]}


def test_controlled_dom_model_primary_uses_cv_winner_and_secondary_history(controlled_dom_result):
    model=controlled_dom_result["model"]
    assert "Nominal logistic" in model["stats"]
    assert "Historical test winner" not in model["stats"]
    assert "0.400" in model["stats"] and "0.700" in model["stats"]
    assert model["counts"]==[[1,1,0],[0,0,0],[0,1,3]]
    assert model["share"]==[[.5,.5,0],[0,0,0],[0,.25,.75]]
    assert model["details"]["history"]==[.99]
    assert sorted(model["details"]["cv"])==[.6,.7]
    assert "not an untouched external test" in model["caveat"]


def test_controlled_dom_live_source_and_cap_switches_cluster_and_evidence_dialog(controlled_dom_result):
    result=controlled_dom_result
    assert result["cachedDate"]=="2026-10-14"
    assert result["cachedBadge"]=="Cached · 2026-10-14"
    assert result["liveAdzuna"]["traces"][0]["y"]==[1,1,2]
    assert result["liveMuse"]["share"]==[.5]
    assert result["liveMuse"]["capHidden"] is False
    assert result["liveCapped"]["share"]==[.25]
    assert "0 / 1" in result["liveCapped"]["stats"]
    assert "Confidence 75.0%" in result["liveRuleDetail"]
    assert result["cluster"]=={"role":1,"name":"Role one"}
    assert result["dialogOpened"] is True and result["dialogClosed"] is True


def test_emitted_product_page_inlines_assets_and_escapes_embedded_json():
    """Render synthetic dictionaries only; do not read canonical data or execute a browser."""
    import sys
    script="""from scripts.build_dashboard import render_product_page
print(render_product_page({'probe': '</script><script src=\"https://bad.invalid/x.js\"></script>'}, {'sources': {}}))
"""
    env={**os.environ,"PYTHONPATH":str(ROOT),"PYTHONDONTWRITEBYTECODE":"1"}
    result=subprocess.run([sys.executable,"-B","-c",script],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20,check=False)
    assert result.returncode==0,result.stderr
    inspector=ProductTemplateInspector();inspector.feed(result.stdout)
    tabs=[attrs for tag,attrs in inspector.elements if attrs.get("role")=="tab"]
    assert [attrs.get("aria-controls") for attrs in tabs]==["overview","skills","salary","clusters","live","model"]
    assert sum(attrs.get("aria-selected")=="true" for attrs in tabs)==1
    assert all(attrs.get("id")=="tab-"+attrs["aria-controls"] for attrs in tabs)
    ids=[attrs["id"] for _,attrs in inspector.elements if attrs.get("id")]
    assert len(ids)==len(set(ids))
    assert "skillmap-product-data" in ids and "skillmap-provenance" in ids
    assert all(not attrs.get("src") for tag,attrs in inspector.elements if tag=="script")
    assert "\\u003c/script\\u003e" in result.stdout
    assert "__NAV__" not in result.stdout and "__PLOTLY__" not in result.stdout
    assert all(1<=len(charts)<=3 for charts in inspector.primary.values())


def relative_luminance(hex_color):
    value=hex_color.removeprefix('#')
    if len(value)==3:
        value=''.join(char*2 for char in value)
    rgb=[int(value[i:i+2],16)/255 for i in (0,2,4)]
    linear=[channel/12.92 if channel<=.04045 else ((channel+.055)/1.055)**2.4 for channel in rgb]
    return sum(channel*weight for channel,weight in zip(linear,(.2126,.7152,.0722)))


def contrast_ratio(foreground,background):
    first,second=sorted([relative_luminance(foreground),relative_luminance(background)])
    return (second+.05)/(first+.05)


def test_small_text_colors_meet_wcag_aa_contrast_in_actual_asset_tokens():
    """Numerical source-color check, not a claim about browser accessibility conformance."""
    import re
    css=(ASSETS/'product.css').read_text()
    tokens=dict(re.findall(r'--([\w-]+)\s*:\s*(#[\da-fA-F]{3,8})(?=\s*[;}])',css))
    for foreground in ('ink','muted','soft'):
        for background in ('surface','canvas'):
            assert contrast_ratio(tokens[foreground],tokens[background])>=4.5,(foreground,background)
    def property_value(selector,property_name):
        matches=[]
        for names,body in re.findall(r'([^{}]+)\{([^{}]+)\}',css):
            if selector in [name.strip() for name in names.split(',')]:
                match=re.search(r'(?:^|;)\s*'+re.escape(property_name)+r'\s*:\s*(#[\da-fA-F]{3,8})(?=\s*[;}]|$)',body)
                if match:
                    matches.append(match.group(1))
        assert matches,(selector,property_name)
        return matches[-1]
    for selector in ('.stat.accent .stat-label','.stat.accent .stat-foot'):
        assert contrast_ratio(property_value(selector,'color'),tokens['accent'])>=4.5
    assert contrast_ratio(property_value('.badge.teal','color'),property_value('.badge.teal','background'))>=4.5
    js=(ASSETS/'product.js').read_text()
    chart_muted=re.search(r"\bmuted\s*=\s*'(#[\da-fA-F]{6})'",js).group(1)
    assert contrast_ratio(chart_muted,tokens['surface'])>=4.5
