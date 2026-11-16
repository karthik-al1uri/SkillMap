"""Statically verify SkillMap's self-contained analyst dashboard without a browser or network."""
import argparse
from collections import Counter
from datetime import date
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re

TABS = {"overview": "Overview", "skills": "Skills", "salary": "Salary", "clusters": "Clusters", "live": "Live", "model": "Model"}
ASSET_TAGS = {"script", "img", "iframe", "video", "audio", "source", "track", "embed", "object", "input"}


def embedded_reference(value):
    return value.strip().lower().startswith(("data:", "#"))


class DashboardParser(HTMLParser):
    """Capture structural roles, inline scripts and resource references only."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = Counter()
        self.tabs = []
        self.panels = []
        self.charts = []
        self.errors = []
        self.stack = []
        self.tab_capture = None
        self.script = None
        self.scripts = []
        self.style = None
        self.styles = []
        self.tablist_count = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        identifier = attributes.get("id")
        if identifier:
            self.ids[identifier] += 1
        role = attributes.get("role")
        if role == "tablist":
            self.tablist_count += 1
        if role == "tab":
            record = {"tag": tag, "attrs": attributes, "text": []}
            self.tabs.append(record)
            self.tab_capture = record
        if role == "tabpanel":
            self.panels.append(attributes)
        classes = set(attributes.get("class", "").split())
        if classes & {"plotly-graph-div", "js-plotly-plot"}:
            panel = next((a.get("id") for _,a in reversed(self.stack) if a.get("role") == "tabpanel"), None)
            self.charts.append({"id": identifier, "panel": panel})
        if tag == "script":
            self.script = {"attrs": attributes, "parts": []}
            if attributes.get("src"):
                self.errors.append("Script src is forbidden; embed JavaScript inline")
        if tag == "style":
            self.style = []
        if tag in ASSET_TAGS:
            for key in ("src", "poster", "data"):
                value = attributes.get(key)
                if value and not embedded_reference(value):
                    self.errors.append(f"Nonembedded resource: {tag}[{key}]={value}")
            if attributes.get("srcset"):
                self.errors.append("srcset requires unsupported resource resolution")
        if tag == "link":
            href = attributes.get("href")
            if href and ("stylesheet" in attributes.get("rel", "").lower().split() or not embedded_reference(href)):
                self.errors.append(f"Linked asset is forbidden: {href}")
        if "style" in attributes:
            self.styles.append(attributes["style"])
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append((tag, attributes))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if self.stack and self.stack[-1][0] == tag:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "script" and self.script is not None:
            self.scripts.append(self.script)
            self.script = None
        if tag == "style" and self.style is not None:
            self.styles.append("".join(self.style))
            self.style = None
        if self.tab_capture is not None and self.tab_capture["tag"] == tag:
            self.tab_capture = None
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if self.script is not None:
            self.script["parts"].append(data)
        elif self.style is not None:
            self.style.append(data)
        elif self.tab_capture is not None:
            self.tab_capture["text"].append(data)


def is_number(value):
    return isinstance(value, (int,float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_metadata(metadata):
    errors = []
    if not isinstance(metadata, dict):
        return ["Provenance must be a JSON object"]
    if metadata.get("schema") != "skillmap.dashboard.v1":
        errors.append("Provenance schema must be skillmap.dashboard.v1")
    sources = metadata.get("sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("sources must be a nonempty registry")
        sources = {}
    for key, value in sources.items():
        if not isinstance(key, str) or not isinstance(value, dict) or not isinstance(value.get("status"), str) or not value.get("status"):
            errors.append(f"Invalid source registry entry: {key}")
        if isinstance(value, dict) and value.get("sha256") is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", str(value["sha256"])):
            errors.append(f"Invalid source SHA-256: {key}")
        if isinstance(value, dict) and value.get("bytes") is not None and (not isinstance(value["bytes"],int) or isinstance(value["bytes"],bool) or value["bytes"] < 0):
            errors.append(f"Invalid source byte count: {key}")
    def source_references(node, location="root"):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "source_ids":
                    if not isinstance(value, list) or any(not isinstance(item,str) or item not in sources for item in value):
                        errors.append(f"Unresolved or invalid source_ids at {location}")
                else:
                    source_references(value, location + "." + key)
        elif isinstance(node,list):
            for i,value in enumerate(node):
                source_references(value, f"{location}[{i}]")
    source_references(metadata)
    wrappers = {}
    for section in ("metrics", "evaluations"):
        mapping = metadata.get(section)
        if not isinstance(mapping,dict):
            errors.append(f"{section} must be an object")
            continue
        wrappers[section] = mapping
        for key, wrapper in mapping.items():
            if not isinstance(wrapper,dict) or not isinstance(wrapper.get("available"),bool) or not isinstance(wrapper.get("status"),str) or not wrapper.get("status") or not isinstance(wrapper.get("source_ids"),list) or "data" not in wrapper:
                errors.append(f"Invalid {section} wrapper: {key}")
    metrics = wrappers.get("metrics",{})
    jobs = metrics.get("n_jobs")
    if not isinstance(jobs,dict):
        errors.append("metrics.n_jobs wrapper is required")
    elif jobs.get("available") and (not isinstance(jobs.get("data"),int) or isinstance(jobs.get("data"),bool) or jobs["data"] <= 0):
        errors.append("Available n_jobs must be a positive integer")
    provenance = metadata.get("provenance")
    if not isinstance(provenance,dict) or provenance.get("salary_grain") != "unique_job_id":
        errors.append("provenance.salary_grain must be unique_job_id")
    dates = []
    for key in ("snapshot_dates", "live_snapshot_dates"):
        wrapper = metrics.get(key,{})
        if isinstance(wrapper,dict) and wrapper.get("available"):
            if not isinstance(wrapper.get("data"),list):
                errors.append(f"metrics.{key}.data must be a list of dates")
            else:
                dates.extend(wrapper["data"])
    evaluations = wrappers.get("evaluations",{})
    live = evaluations.get("live",{})
    if isinstance(live,dict) and live.get("available"):
        data = live.get("data",{})
        report = data.get("report",{}) if isinstance(data,dict) else {}
        source_status = report.get("source_status",{}) if isinstance(report,dict) else {}
        fetched = source_status.get("fetch_date") if isinstance(source_status,dict) else None
        if fetched:
            dates.append(fetched)
        if not dates:
            errors.append("Available live evaluation requires at least one snapshot/fetch date")
    for value in dates:
        try:
            if not isinstance(value,str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError("not an ISO date")
        except (ValueError,TypeError):
            errors.append(f"Invalid live date: {value!r}")
    grouped = evaluations.get("grouped",{})
    if "grouped" not in evaluations or not isinstance(grouped,dict):
        errors.append("evaluations.grouped must be a wrapper")
    elif grouped.get("available"):
        data = grouped.get("data",{})
        report = data.get("report",{}) if isinstance(data,dict) else {}
        design = data.get("frozen_design",{}) if isinstance(data,dict) else {}
        if not isinstance(report,dict) or report.get("status") != "complete":
            errors.append("Available grouped report must be complete")
            report = {}
        rows = report.get("cv_rows")
        if not isinstance(rows,list) or not rows or any(not isinstance(row,dict) or not isinstance(row.get("candidate"),str) or not is_number(row.get("mean_f1_macro")) or not 0 <= row["mean_f1_macro"] <= 1 for row in rows):
            errors.append("Grouped cv_rows require named candidates and valid mean_f1_macro")
        else:
            names = [row["candidate"] for row in rows]
            if len(names) != len(set(names)):
                errors.append("Grouped candidate names must be unique")
            winner = max(rows,key=lambda row:row["mean_f1_macro"])["candidate"]
            if report.get("winner") != winner:
                errors.append("Grouped winner must follow mean inner CV F1 and candidate-order ties")
        selection = design.get("selection","") if isinstance(design,dict) else ""
        if not isinstance(selection,str) or not any(word in selection.lower() for word in ("inner", "training", "cv", "cross-validation")):
            errors.append("Grouped frozen design must state training/inner CV selection")
        cv_selection = data.get("cv_selection") if isinstance(data,dict) else None
        if isinstance(cv_selection,dict) and cv_selection.get("winner") is not None and cv_selection["winner"] != report.get("winner"):
            errors.append("Grouped cv_selection and report winners disagree")
        outer = report.get("outer_metrics")
        if not isinstance(outer,dict) or not is_number(outer.get("f1_macro")) or not 0 <= outer["f1_macro"] <= 1:
            errors.append("Grouped outer_metrics require valid f1_macro")
        elif any(key in outer and outer[key] is not None and (not is_number(outer[key]) or not 0 <= outer[key] <= 1) for key in ("accuracy","roc_auc_ovr_macro","roc_auc_ovr")):
            errors.append("Grouped outer metrics must be probabilities/scores in [0,1]")
    return errors


def verify_html(text):
    parser = DashboardParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as error:
        parser.errors.append(f"HTML parse failed: {error}")
    errors = list(parser.errors)
    duplicates = sorted(key for key,count in parser.ids.items() if count > 1)
    if duplicates:
        errors.append("Duplicate IDs: " + ", ".join(duplicates))
    if parser.tablist_count != 1:
        errors.append("Exactly one role=tablist is required")
    panels = {panel.get("id"):panel for panel in parser.panels}
    if len(parser.panels) != 6 or set(panels) != set(TABS):
        errors.append("Exactly the six expected role=tabpanel IDs are required")
    observed = []
    selected = 0
    for tab in parser.tabs:
        attrs = tab["attrs"]
        target = attrs.get("aria-controls")
        label = " ".join("".join(tab["text"]).split())
        observed.append(target)
        if tab["tag"] != "button" or not attrs.get("id") or target not in TABS or label != TABS.get(target):
            errors.append(f"Invalid analyst tab: target={target!r}, label={label!r}")
        if attrs.get("data-tab") not in (None,target):
            errors.append(f"Tab data-tab disagrees with aria-controls: {target}")
        if attrs.get("aria-selected") not in ("true","false"):
            errors.append(f"Tab requires aria-selected true/false: {target}")
        selected += attrs.get("aria-selected") == "true"
        panel = panels.get(target,{})
        if panel.get("aria-labelledby") != attrs.get("id"):
            errors.append(f"Panel aria-labelledby must resolve to its tab: {target}")
        if "panel" not in panel.get("class", "").split():
            errors.append(f"Panel requires panel CSS class: {target}")
    if len(parser.tabs) != 6 or set(observed) != set(TABS):
        errors.append("Exactly six expected analyst tabs are required")
    if selected != 1:
        errors.append("Exactly one tab must initially have aria-selected=true")
    if not parser.charts:
        errors.append("At least one Plotly chart container is required")
    for chart in parser.charts:
        if not chart["id"] or chart["panel"] not in TABS:
            errors.append("Every chart requires a unique ID inside a known tabpanel")
    inline_plotly = any(script["attrs"].get("type","").lower() not in ("application/json","application/ld+json") and not script["attrs"].get("src") and re.search(r"Plotly\s*=|plotly\.js", "".join(script["parts"]), re.I) for script in parser.scripts)
    if not inline_plotly:
        errors.append("Inline Plotly JavaScript is required")
    for style in parser.styles:
        if re.search(r"@import\b",style,re.I):
            errors.append("CSS @import is forbidden")
        for match in re.finditer(r"url\(\s*(['\"]?)(.*?)\1\s*\)",style,re.I|re.S):
            if not embedded_reference(match.group(2)):
                errors.append("Nonembedded CSS resource: " + match.group(2))
    provenance_scripts = [script for script in parser.scripts if script["attrs"].get("id") == "skillmap-provenance"]
    metadata = None
    metadata_valid = False
    if len(provenance_scripts) != 1 or provenance_scripts[0]["attrs"].get("type") != "application/json":
        errors.append("Exactly one application/json skillmap-provenance script is required")
    else:
        try:
            metadata = json.loads("".join(provenance_scripts[0]["parts"]), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON number")))
            metadata_errors = validate_metadata(metadata)
            errors.extend(metadata_errors)
            metadata_valid = not metadata_errors
        except (ValueError,TypeError) as error:
            errors.append(f"Invalid provenance JSON: {error}")
    checks = {"six_analyst_tabs":len(parser.tabs)==6 and set(observed)==set(TABS), "six_tabpanels":len(parser.panels)==6 and set(panels)==set(TABS),
              "unique_ids":not duplicates,"inline_plotly":bool(inline_plotly),"provenance_present":metadata is not None,"metadata_schema_consistent":metadata_valid,
              "no_external_required_assets":not any("resource" in error.lower() or "asset" in error.lower() or "script src" in error.lower() or "@import" in error.lower() for error in errors)}
    return {"status":"passed" if not errors else "failed","checks":checks,"errors":errors,"chart_count":len(parser.charts),"panels":sorted(key for key in panels if key is not None),
            "tab_labels":[" ".join("".join(tab["text"]).split()) for tab in parser.tabs],
            "metadata_schema":metadata.get("schema") if isinstance(metadata,dict) else None,
            "limitations":["Static HTML/schema verification only; does not execute JavaScript or prove runtime rendering/network behavior.","Source IDs and supplied hashes are validated structurally; this verifier does not reread registered source artifacts."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args(argv)
    html_path = args.html.resolve(strict=True)
    out = args.out.resolve()
    repository = Path(__file__).resolve().parents[1]
    if out.exists() or out == repository or repository in out.parents or any((parent/"CLAUDE.md").is_file() for parent in (out.parent,*out.parent.parents)):
        parser.error("--out must be fresh and outside repository trees")
    raw = html_path.read_bytes()
    report = {"html":str(html_path),"html_bytes":len(raw),"html_sha256":hashlib.sha256(raw).hexdigest(),**verify_html(raw.decode("utf-8"))}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":report["status"],"chart_count":report["chart_count"],"errors":report["errors"],"report":str(out)}))
    return report


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "passed" else 1)
