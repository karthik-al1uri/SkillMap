"""Dashboard selection uses training-CV metadata, even when test scores disagree."""
import pandas as pd
import pytest
from scripts import build_dashboard as dashboard


@pytest.mark.parametrize("flags", [[True, False], ["True", "False"]])
def test_display_uses_cv_winner_and_its_own_metrics(monkeypatch, flags):
    results = pd.DataFrame({"model": ["cv winner", "test winner"], "f1_macro": [.4, .9],
                            "roc_auc_ovr": [.6, .95], "selected_by_training_cv": flags})
    for name in ("model_scores", "model_tiers"):
        monkeypatch.setattr(dashboard, name, lambda *args: None)
    monkeypatch.setattr(dashboard, "card", lambda *args, **kwargs: "chart")
    data = {key: None for key in ["04_association_rules", "04_high_salary_rules", "06_cluster_labels",
                                 "kmeans_sweep", "08_skill_comparison", "08_rules_validation"]}
    data.update({"05_classification_results": results, "snapshots": {}, "evaluations": {"grouped": {"report": {
        "status": "complete", "winner": "cv winner", "cv_rows": [
            {"candidate": "cv winner", "mean_f1_macro": .8}, {"candidate": "test winner", "mean_f1_macro": .7}],
        "outer_metrics": {"f1_macro": .4, "roc_auc_ovr_macro": .6,
                          "f1_per_class": {"Low": .4, "Mid": .4, "High": .4},
                          "confusion_matrix": [[1,0,0],[0,1,0],[0,0,1]]}}}}})
    models = next(tab for tab in dashboard.build_tabs(data) if tab[0] == "model")
    tiles = "".join(models[2])
    assert "cv winner" in tiles and "test winner" not in tiles
    assert "0.400" in tiles and "0.600" in tiles
    assert "0.900" not in tiles and "0.950" not in tiles


@pytest.mark.parametrize("flags", [None, [False, False], [True, True]])
def test_missing_or_ambiguous_selection_never_falls_back_to_test_f1(flags):
    results = pd.DataFrame({"model": ["a", "b"], "f1_macro": [.4, .9]})
    if flags is not None:
        results["selected_by_training_cv"] = flags
    assert dashboard.selected_model_row(results) is None


def test_association_views_limit_and_strict_technical_filter():
    rules = pd.DataFrame([{'antecedents': f's{i}', 'consequents': 'target', 'support': .1,
                           'confidence': .6, 'lift': 2. + i/10, 'high_tier_rate': .5 + i/100,
                           'jobs_with_itemset': 10+i, 'high_tier_lift': 1.5} for i in range(20)])
    assert len(dashboard.rules_by_lift(rules).data[0].x) == 10
    jobs = pd.DataFrame({'salary_tier': ['High', 'Low']})
    assert len(dashboard.rules_by_high_share(rules, jobs).data[0].x) == 5
    mixed = pd.DataFrame([{'antecedents': 'python', 'consequents': 'sql'},
                          {'antecedents': 'python', 'consequents': 'development'}])
    assert dashboard.strict_technical(mixed).consequents.tolist() == ['sql']


def facts_fixture():
    def entry(data, status='artifact_derived'):
        return {'available': True, 'status': status, 'source_ids': ['saved'], 'data': data}
    jobs = [{'job_id': 1, 'normalized_salary': 10000., 'salary_tier': 'Low', 'matched_skills': ['python']},
            {'job_id': 2, 'normalized_salary': 30000., 'salary_tier': 'Low', 'matched_skills': []},
            {'job_id': 3, 'normalized_salary': 100000., 'salary_tier': 'Mid', 'matched_skills': ['python']}]
    return {'schema': 'skillmap.dashboard.v1', 'sources': {'saved': {'path': 'fixture', 'sha256': 'a'*64, 'status': 'artifact_derived'}},
            'provenance': {'salary_grain': 'unique_job_id'}, 'evaluations': {},
            'metrics': {'n_jobs': entry(3), 'n_skills_vocab': entry(1), 'skill_counts': entry({'python':2}), 'median_salary': entry(30000.)},
            'tables': {'jobs': entry(jobs)}}


def test_salary_statistics_are_job_level_not_skill_weighted():
    jobs = pd.DataFrame(facts_fixture()['tables']['jobs']['data'])
    assert dashboard.salary_summary(jobs) == {'count': 3, 'median': 30000.}
    with pytest.raises(ValueError, match='unique job_id'):
        dashboard.salary_summary(pd.concat([jobs, jobs.iloc[[2]]]))


def test_generated_facts_are_sole_data_input_and_html_has_six_accessible_views(tmp_path, monkeypatch):
    import json
    from html.parser import HTMLParser
    bundle = facts_fixture()
    # Unsafe closing tags in source metadata must remain escaped in embedded JSON.
    bundle['sources']['saved']['path'] = '</script><script>unexpected()</script>'
    path = tmp_path/'facts.json';path.write_text(json.dumps(bundle))
    monkeypatch.setattr(dashboard, 'get_plotlyjs', lambda: 'window.Plotly={};')
    data = dashboard.load_inputs(path)
    tabs = dashboard.build_tabs(data)
    assert [(tab[0],tab[1]) for tab in tabs] == [('overview','Overview'),('skills','Skills'),('salary','Salary'),
        ('clusters','Clusters'),('live','Live'),('model','Model')]
    page = dashboard.render_page(data)
    assert 'aria-labelledby="tab-salary"' in page
    assert 'id="skillmap-provenance"' in page
    assert '</script><script>unexpected()' not in page
    assert 'No complete grouped evaluation supplied' in page or 'No complete grouped evaluation supplied' in page.replace('\\u003c','<')
    class Tags(HTMLParser):
        def __init__(self):super().__init__();self.tabs=[]
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if tag=='button' and attrs.get('role')=='tab':self.tabs.append(attrs)
    tags=Tags();tags.feed(page)
    assert len(tags.tabs)==6 and sum(t['aria-selected']=='true' for t in tags.tabs)==1


def test_missing_or_inconsistent_provenance_fails_closed(tmp_path):
    import json
    bundle=facts_fixture();bundle['tables']['jobs']['source_ids']=['unknown']
    path=tmp_path/'bad.json';path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match='Unknown provenance'):
        dashboard.load_inputs(path)
    with pytest.raises(ValueError, match='training CV'):
        dashboard.grouped_report({'evaluations': {'grouped': {'report': {'status':'complete','winner':'test winner',
            'cv_rows':[{'candidate':'cv winner','mean_f1_macro':.8},{'candidate':'test winner','mean_f1_macro':.7}]}}}})


def test_output_guards_preserve_existing_files(tmp_path):
    existing=tmp_path/'existing.html';existing.write_text('preserved')
    for destination in (existing, dashboard.ROOT/'outputs'/'unwanted.html'):
        with pytest.raises(SystemExit):
            dashboard.main(['--facts',str(tmp_path/'missing.json'),'--out',str(destination)])
    assert existing.read_text()=='preserved'



def test_empty_technical_view_keeps_zero_result_instead_of_losing_schema():
    data = {"04_technical_rules_low_support": pd.DataFrame(), "metrics": {}, "evaluations": {}}
    skills = next(tab for tab in dashboard.build_tabs(data) if tab[0] == "skills")
    assert "technical-only rules" in "".join(skills[2])
    assert "exclusively forced-technical" in "".join(skills[3])


def test_tab_controller_keyboard_navigation_and_legacy_links_without_browser(tmp_path):
    import json
    import shutil
    import subprocess
    node = shutil.which('node')
    if node is None:
        pytest.skip('Optional Node-based controller simulation unavailable; no browser execution claimed')
    harness = r'''
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ids = ['overview','skills','salary','clusters','live','model'];
function run(hash, stored) {
  const panels = ids.map(id=>({id, hidden: false}));
  const buttons = ids.map(id=>({dataset:{tab:id}, attrs:{}, handlers:{},
    setAttribute(k,v){this.attrs[k]=String(v)}, addEventListener(k,f){this.handlers[k]=f}, focus(){this.focused=true}}));
  const context = {JSON, Map, Set, history:{replaceState(a,b,url){this.url=url}},
    location:{hash}, localStorage:{getItem(){return stored},setItem(k,v){this.value=v}},
    document:{documentElement:{dataset:{}}, querySelectorAll(selector){
      if(selector==='nav button') return buttons;
      if(selector==='.panel') return panels;
      return [];
    }, querySelector(){return buttons[0]}},
    matchMedia(){return {matches:false,addEventListener(){}}},
    MutationObserver:class {constructor(fn){} observe(){}},
    Plotly:{react(){},Plots:{resize(){}}}};
  vm.runInNewContext(SOURCE,context);
  assert.equal(buttons.filter(b=>b.attrs['aria-selected']==='true').length,1);
  assert.equal(panels.filter(p=>!p.hidden).length,1);
  assert.equal(buttons.filter(b=>b.attrs.tabindex==='0').length,1);
  return {buttons,panels,context};
}
let state=run('#models',null);
assert.equal(state.panels.find(p=>!p.hidden).id,'model');
state.buttons[0].handlers.click();
assert.equal(state.panels.find(p=>!p.hidden).id,'overview');
state.buttons[0].handlers.keydown({key:'ArrowRight',preventDefault(){}});
assert.equal(state.panels.find(p=>!p.hidden).id,'skills');
state.buttons[1].handlers.keydown({key:'End',preventDefault(){}});
assert.equal(state.panels.find(p=>!p.hidden).id,'model');
state.buttons[5].handlers.keydown({key:'Home',preventDefault(){}});
assert.equal(state.panels.find(p=>!p.hidden).id,'overview');
assert.equal(run('#unrelated',null).panels.find(p=>!p.hidden).id,'overview');
assert.equal(run('', 'rules').panels.find(p=>!p.hidden).id,'skills');
console.log('Controlled tab/keyboard/legacy-link checks passed; not browser rendering.');
'''
    script = tmp_path/'controller.js'
    script.write_text('const SOURCE = '+json.dumps(dashboard.PAGE_JS.replace('__MAP__','{}'))+';\n'+harness)
    result=subprocess.run([node,str(script)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0, result.stderr
    assert 'checks passed' in result.stdout
