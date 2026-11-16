"""Small synthetic pages test offline dashboard structure and provenance semantics."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts.verify_dashboard import TABS, main, validate_metadata, verify_html


def wrapper(data, available=True):
    return {"available":available,"status":"artifact_derived" if available else "missing", "source_ids":["jobs"] if available else [],"data":data}


def metadata():
    return {"schema":"skillmap.dashboard.v1","sources":{"jobs":{"path":"saved.csv","bytes":10,"sha256":"a"*64,"status":"historical_exploratory"}},
        "metrics":{"n_jobs":wrapper(100),"median_salary":wrapper(80000),"snapshot_dates":wrapper(["2026-10-14"])},
        "evaluations":{"grouped":wrapper({"report":{"status":"complete","winner":"cv_winner","cv_rows":[{"candidate":"cv_winner","mean_f1_macro":.7,"test_f1":.4},{"candidate":"test_winner","mean_f1_macro":.6,"test_f1":.9}],"outer_metrics":{"f1_macro":.4,"accuracy":.5,"roc_auc_ovr_macro":.6}},"frozen_design":{"selection":"maximum mean inner CV F1"},"cv_selection":{"winner":"cv_winner"}}),"live":wrapper({"report":{"source_status":{"fetch_date":"2026-10-14"}}})},
        "provenance":{"salary_grain":"unique_job_id","source_ids":["jobs"]},"table_provenance":{"salary":{"source_ids":["jobs"]}}}


def page(payload=None):
    payload=metadata() if payload is None else payload
    tabs="".join(f'<button id="tab-{key}" role="tab" data-tab="{key}" aria-controls="{key}" aria-selected="{str(i==0).lower()}">{label}</button>' for i,(key,label) in enumerate(TABS.items()))
    panels="".join(f'<section id="{key}" role="tabpanel" class="panel" aria-labelledby="tab-{key}">'+ ('<div id="chart-one" class="plotly-graph-div"></div>' if key=="skills" else '')+'</section>' for key in TABS)
    return '<!doctype html><html><head><style>.panel {display:block}</style><script>window.Plotly = {newPlot:function(){}};</script></head><body><nav role="tablist">'+tabs+'</nav>'+panels+'<script type="application/json" id="skillmap-provenance">'+json.dumps(payload)+'</script></body></html>'


def test_valid_page_uses_cv_metadata_instead_of_higher_test_f1():
    result=verify_html(page())
    assert result["status"]=="passed",result["errors"]
    assert result["chart_count"]==1
    assert set(result["panels"])==set(TABS)
    assert all(result["checks"].values())


@pytest.mark.parametrize("mutation,expected",[
    (lambda text:text.replace('role="tab"','role="presentation"',1),'six expected analyst tabs'),
    (lambda text:text.replace('id="chart-one"','id="overview"'),'Duplicate IDs'),
    (lambda text:text.replace('</head>','<script src="https://example.com/plotly.js"></script></head>'),'Script src'),
    (lambda text:text.replace('</head>','<link rel="stylesheet" href="style.css"></head>'),'Linked asset'),
    (lambda text:text.replace('id="skillmap-provenance"','id="other-json"'),'skillmap-provenance'),
    (lambda text:text.replace('aria-labelledby="tab-salary"','aria-labelledby="tab-model"'),'aria-labelledby'),
    (lambda text:text.replace('window.Plotly = {newPlot:function(){}};', 'Plotly.newPlot("chart-one",[]);'),'Inline Plotly'),
    (lambda text:text.replace('.panel {display:block}', '.panel {background:url(https://example.com/image.png)}'),'Nonembedded CSS'),
])
def test_structural_failures(mutation,expected):
    report=verify_html(mutation(page()))
    assert report["status"]=="failed"
    assert any(expected in error for error in report["errors"]),report["errors"]


@pytest.mark.parametrize("mutation,expected",[
    (lambda data:data["metrics"]["n_jobs"].update(data=0),'positive integer'),
    (lambda data:data["provenance"].update(salary_grain="job_skill_fact"),'unique_job_id'),
    (lambda data:data["evaluations"]["grouped"]["data"]["report"].update(winner="test_winner"),'mean inner CV'),
    (lambda data:data["metrics"]["snapshot_dates"].update(data=["2026-02-30"]),'Invalid live date'),
    (lambda data:data["table_provenance"]["salary"].update(source_ids=["missing"]),'Unresolved'),
])
def test_metadata_consistency_failures(mutation,expected):
    payload=metadata();mutation(payload)
    assert any(expected in error for error in validate_metadata(payload))


def test_explicit_unavailable_evaluations_are_valid_without_invented_values():
    payload=metadata()
    payload["evaluations"]["grouped"]=wrapper(None,False)
    payload["evaluations"]["live"]=wrapper(None,False)
    payload["metrics"]["snapshot_dates"]=wrapper(None,False)
    assert verify_html(page(payload))["status"]=="passed"


def test_cli_hashes_source_writes_fresh_report_and_refuses_overwrite(tmp_path):
    html_path=tmp_path / "dashboard.html";html_path.write_text(page())
    out=tmp_path / "verification.json"
    report=main(["--html",str(html_path),"--out",str(out)])
    assert report["status"]=="passed"
    assert report["html_sha256"]==hashlib.sha256(html_path.read_bytes()).hexdigest()
    before=out.read_bytes()
    with pytest.raises(SystemExit):
        main(["--html",str(html_path),"--out",str(out)])
    assert out.read_bytes()==before
    root=tmp_path / "canonical";root.mkdir();(root / "CLAUDE.md").write_text("marker")
    with pytest.raises(SystemExit):
        main(["--html",str(html_path),"--out",str(root / "new.json")])
    assert not (root / "new.json").exists()
