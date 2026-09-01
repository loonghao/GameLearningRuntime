use std::fs;
use std::path::{Component, Path};

use serde_json::json;

use crate::commands::CLI_OUTPUT_SCHEMA_VERSION;
use crate::contracts::sha256_file;
use crate::error::{Error, Result};
use crate::process::relative_portable;
use crate::project::Project;
use crate::store::Store;

pub const RUN_REPORT_SCHEMA_VERSION: &str = "glr.run-report.v1";

/// Build an offline, data-only report for one persisted run.
pub fn build(
    project: &Project,
    store: &Store,
    run_id: &str,
    output: Option<&Path>,
    as_json: bool,
) -> Result<i32> {
    let run = store.get_run(run_id)?;
    let run_dir = project.data_dir.join("runs").join(run_id);
    if !run_dir.is_dir() || run_dir.is_symlink() {
        return Err(Error::Missing(run_dir));
    }
    let output_dir = match output {
        Some(path) => {
            let resolved = if path.is_absolute() {
                path.to_path_buf()
            } else {
                run_dir.join(path)
            };
            ensure_portable_child(&run_dir, &resolved, "report output")?;
            resolved
        }
        None => run_dir.join("report"),
    };
    ensure_no_symlink_components(&run_dir, &output_dir, "report output")?;
    fs::create_dir_all(&output_dir)?;

    let artifacts = store
        .list_artifacts(run_id)?
        .into_iter()
        .map(|artifact| {
            let source = artifact_source(&run_dir, &artifact.path)?;
            if source.metadata()?.len() != artifact.size_bytes
                || sha256_file(&source)? != artifact.sha256
            {
                return Err(Error::Contract(format!(
                    "run artifact failed verification: {}",
                    artifact.path
                )));
            }
            let href = relative_href(&run_dir, &output_dir, &source)?;
            Ok(json!({
                "path": artifact.path,
                "role": artifact.role,
                "media_type": artifact.media_type,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "href": href,
                "metadata": artifact.metadata,
            }))
        })
        .collect::<Result<Vec<_>>>()?;

    let data = json!({
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "run": run,
        "events": store.list_events(run_id)?,
        "metrics": store.list_metrics(run_id)?,
        "artifacts": artifacts,
    });
    let data_json = serde_json::to_string(&data)?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('/', "\\u002f");
    let html = render_html(&data_json);
    let target = output_dir.join("index.html");
    write_report(&target, html.as_bytes())?;
    let relative = relative_portable(&run_dir, &target)?;
    let artifact =
        store.register_artifact(run_id, &relative, &target, "run-report", "text/html")?;

    let payload = json!({
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "path": relative,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    });
    let envelope = json!({
        "schema_version": CLI_OUTPUT_SCHEMA_VERSION,
        "command": "report.build",
        "data": payload,
    });
    if as_json {
        println!("{}", serde_json::to_string(&envelope)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&envelope)?);
    }
    Ok(0)
}

fn relative_href(run_dir: &Path, output_dir: &Path, source: &Path) -> Result<String> {
    let source_relative = source
        .strip_prefix(run_dir)
        .map_err(|_| Error::Invalid("report artifact has no relative path".into()))?;
    let output_relative = output_dir
        .strip_prefix(run_dir)
        .map_err(|_| Error::Invalid("report output has no relative path".into()))?;
    let depth = output_relative.components().count();
    let prefix = std::iter::repeat_n("..".to_string(), depth).collect::<Vec<_>>();
    let mut parts: Vec<String> = prefix;
    parts.extend(
        source_relative
            .iter()
            .map(|part| part.to_string_lossy().into_owned()),
    );
    Ok(parts.join("/"))
}

fn ensure_portable_child(root: &Path, path: &Path, label: &str) -> Result<String> {
    let relative = relative_portable(root, path)?;
    let relative_path = Path::new(&relative);
    if relative.is_empty()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::Invalid(format!(
            "{label} must stay inside the run directory"
        )));
    }
    Ok(relative)
}

fn ensure_no_symlink_components(root: &Path, path: &Path, label: &str) -> Result<()> {
    let relative = ensure_portable_child(root, path, label)?;
    let mut current = root.to_path_buf();
    for component in Path::new(&relative).components() {
        let Component::Normal(part) = component else {
            return Err(Error::Invalid(format!(
                "{label} must use portable path components"
            )));
        };
        current.push(part);
        if current.is_symlink() {
            return Err(Error::Invalid(format!("{label} cannot traverse a symlink")));
        }
    }
    Ok(())
}

fn artifact_source(run_dir: &Path, relative_path: &str) -> Result<std::path::PathBuf> {
    let relative = Path::new(relative_path);
    if relative_path.is_empty()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::Invalid(format!(
            "run artifact path is not portable: {relative_path}"
        )));
    }
    let source = run_dir.join(relative);
    if source.is_symlink() || !source.is_file() {
        return Err(Error::Missing(source));
    }
    let canonical_run_dir = run_dir.canonicalize()?;
    let canonical_source = source.canonicalize()?;
    if !canonical_source.starts_with(&canonical_run_dir) {
        return Err(Error::Invalid(format!(
            "run artifact escapes the run directory: {relative_path}"
        )));
    }
    Ok(source)
}

fn write_report(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::Invalid("report output has no parent".into()))?;
    if path.is_symlink() {
        return Err(Error::Invalid("report output cannot be a symlink".into()));
    }
    let temporary = tempfile::NamedTempFile::new_in(parent)?;
    fs::write(temporary.path(), bytes)?;
    fs::copy(temporary.path(), path)?;
    Ok(())
}

fn render_html(data_json: &str) -> String {
    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self' data:; media-src 'self';">
<title>GLR run report</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --line:#273454; --text:#e8eefc; --muted:#94a3c7; --accent:#74d4ff; --good:#7ee2a8; --bad:#ff8e9e; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at top right,#17294b 0,#0b1020 42%); color:var(--text); font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
.shell {{ max-width:1200px; margin:0 auto; padding:32px 20px 56px; }} header {{ display:flex; justify-content:space-between; gap:20px; align-items:flex-end; margin-bottom:24px; }} h1 {{ margin:0; font-size:clamp(25px,4vw,42px); letter-spacing:-.03em; }} h2 {{ margin:0 0 14px; font-size:18px; }} .eyebrow {{ color:var(--accent); text-transform:uppercase; letter-spacing:.14em; font-size:11px; font-weight:700; }} .muted {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(155px,1fr)); gap:12px; margin-bottom:22px; }} .card,.panel {{ background:rgba(18,26,45,.88); border:1px solid var(--line); border-radius:16px; box-shadow:0 12px 30px #05081266; }} .card {{ padding:16px; }} .value {{ display:block; font-size:26px; font-weight:750; margin-top:4px; }}
.tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }} button {{ border:1px solid var(--line); border-radius:999px; padding:9px 14px; color:var(--muted); background:#0e1629; cursor:pointer; }} button[aria-selected="true"] {{ color:#06101e; background:var(--accent); border-color:var(--accent); font-weight:700; }} .panel {{ display:none; padding:18px; }} .panel.active {{ display:block; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:9px 8px; border-bottom:1px solid #22304b; vertical-align:top; }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }} td code {{ color:#b9c7e6; font-size:12px; }} .pill {{ display:inline-block; border-radius:999px; padding:2px 8px; background:#203451; color:var(--accent); font-size:12px; }} .good {{ color:var(--good); }} .bad {{ color:var(--bad); }}
.metric {{ display:grid; grid-template-columns:180px 1fr 90px; gap:12px; align-items:center; margin:10px 0; }} .bar {{ height:9px; border-radius:99px; background:#202d49; overflow:hidden; }} .bar > i {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),#9a8cff); border-radius:inherit; }} .route {{ width:100%; min-height:280px; border:1px solid var(--line); border-radius:12px; background:#0b1325; }} .route polyline {{ fill:none; stroke:var(--accent); stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; }} .route circle {{ fill:var(--good); stroke:#08101f; stroke-width:2; }} .empty {{ padding:28px 8px; text-align:center; color:var(--muted); }} a {{ color:var(--accent); }} .media-preview {{ display:block; max-width:min(280px,100%); max-height:180px; margin-top:8px; border:1px solid var(--line); border-radius:8px; object-fit:contain; background:#0b1325; }}
@media (max-width:700px) {{ header {{ display:block; }} .metric {{ grid-template-columns:1fr; gap:4px; }} th:nth-child(4),td:nth-child(4) {{ display:none; }} }}
</style>
</head>
<body>
<main class="shell">
<header><div><div class="eyebrow">Game Learning Runtime · offline evidence</div><h1 id="title">Run report</h1><div id="subtitle" class="muted"></div></div><div id="status" class="pill"></div></header>
<section id="summary" class="grid"></section>
<nav class="tabs" role="tablist" aria-label="Report sections"><button aria-selected="true" data-panel="overview">Overview</button><button aria-selected="false" data-panel="timeline">Timeline</button><button aria-selected="false" data-panel="route">Route</button><button aria-selected="false" data-panel="progression">Progression</button><button aria-selected="false" data-panel="matches">Matches</button><button aria-selected="false" data-panel="artifacts">Artifacts</button></nav>
<section id="overview" class="panel active"><h2>Training metrics</h2><div id="metrics"></div></section>
<section id="timeline" class="panel"><h2>Event timeline</h2><input id="filter" aria-label="Filter events" placeholder="Filter by event kind" style="width:100%;padding:10px;border-radius:9px;border:1px solid var(--line);background:#0c1425;color:var(--text);margin-bottom:12px"><div id="events"></div></section>
<section id="route" class="panel"><h2>Route trace</h2><p class="muted">Samples are rendered only when an adapter emitted finite <code>position</code> coordinates. The report never infers a route from pixels.</p><svg class="route" id="route-svg" viewBox="0 0 800 360" role="img" aria-label="Route trace"></svg><div id="route-caption" class="muted" style="margin-top:10px"></div></section>
<section id="progression" class="panel"><h2>Unlocks and progression</h2><div id="progression-table"></div></section>
<section id="matches" class="panel"><h2>Matches</h2><div id="matches-table"></div></section>
<section id="artifacts" class="panel"><h2>Checksummed artifacts</h2><div id="artifact-table"></div></section>
</main>
<script id="glr-report-data" type="application/json">{data}</script>
<script>
(() => {{
  const data = JSON.parse(document.getElementById('glr-report-data').textContent);
  const run = data.run, events = data.events || [], metrics = data.metrics || [], artifacts = data.artifacts || [];
  const text = (value) => value == null ? '' : String(value);
  const cell = (value) => {{ const node = document.createElement('td'); node.textContent = text(value); return node; }};
  const empty = (target, message) => {{ target.textContent = message; target.className = 'empty'; }};
  document.getElementById('title').textContent = run.run_id;
  document.getElementById('subtitle').textContent = `${{run.environment_id}} · ${{run.kind}} · ${{data.schema_version}}`;
  const status = document.getElementById('status'); status.textContent = run.status; status.classList.add(run.status === 'succeeded' ? 'good' : 'bad');
  const episodes = new Set(events.map(e => e.episode_id).filter(Boolean));
  const cards = [['Episodes', episodes.size], ['Events', events.length], ['Metrics', metrics.length], ['Artifacts', artifacts.length]];
  const summary = document.getElementById('summary'); cards.forEach(([label, value]) => {{ const card = document.createElement('div'); card.className='card'; const caption=document.createElement('span'); caption.className='muted'; caption.textContent=label; const number=document.createElement('span'); number.className='value'; number.textContent=value; card.append(caption,number); summary.append(card); }});
  const metricBox = document.getElementById('metrics');
  const groups = new Map(); metrics.forEach(metric => {{ if (!groups.has(metric.name)) groups.set(metric.name, []); groups.get(metric.name).push(metric); }});
  if (!groups.size) empty(metricBox, 'No scalar metrics were recorded.');
  for (const [name, values] of groups) {{ const row=document.createElement('div'); row.className='metric'; const label=document.createElement('code'); label.textContent=name; const bar=document.createElement('div'); bar.className='bar'; const fill=document.createElement('i'); const max=Math.max(...values.map(v=>Math.abs(Number(v.value))||0),1); fill.style.width=`${{Math.min(100,Math.abs(Number(values.at(-1).value))/max*100)}}%`; bar.append(fill); const latest=document.createElement('span'); latest.textContent=Number(values.at(-1).value).toPrecision(5); row.append(label,bar,latest); metricBox.append(row); }}
  const renderEvents = () => {{ const target=document.getElementById('events'); target.replaceChildren(); const query=document.getElementById('filter').value.toLowerCase(); const selected=events.filter(e=>!query || e.kind.toLowerCase().includes(query)); if(!selected.length) return empty(target,'No matching events.'); const table=document.createElement('table'), head=document.createElement('tr'); ['Sequence','Step','Kind','Payload'].forEach(value=>{{const th=document.createElement('th');th.textContent=value;head.append(th);}}); table.append(head); selected.forEach(e=>{{const row=document.createElement('tr'); row.append(cell(e.sequence_id),cell(e.step_id),cell(e.kind),cell(JSON.stringify(e.payload))); table.append(row);}}); target.append(table); }}; document.getElementById('filter').addEventListener('input',renderEvents); renderEvents();
  const routeEvents=events.filter(e=>e.kind.startsWith('navigation.route')); const points=routeEvents.map(e=>Array.isArray(e.payload?.position)?e.payload.position:e.payload?.xyz).filter(p=>Array.isArray(p)&&p.length>=2&&p.every(Number.isFinite)); const svg=document.getElementById('route-svg'); if(!points.length) {{ empty(svg.parentElement,'No route samples were recorded.'); }} else {{ const xs=points.map(p=>p[0]), ys=points.map(p=>p[1]), minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys), sx=x=>40+(x-minX)/Math.max(maxX-minX,1)*720, sy=y=>320-(y-minY)/Math.max(maxY-minY,1)*280; const line=document.createElementNS('http://www.w3.org/2000/svg','polyline'); line.setAttribute('points',points.map(p=>`${{sx(p[0])}},${{sy(p[1])}}`).join(' ')); svg.append(line); points.forEach((p,i)=>{{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',sx(p[0]));c.setAttribute('cy',sy(p[1]));c.setAttribute('r',i===0||i===points.length-1?'6':'3');svg.append(c);}}); document.getElementById('route-caption').textContent=`${{points.length}} samples · x ${{minX.toFixed(2)}}–${{maxX.toFixed(2)}} · y ${{minY.toFixed(2)}}–${{maxY.toFixed(2)}}`; }}
  const progression=events.filter(e=>e.kind.startsWith('progression.')); const progressionTarget=document.getElementById('progression-table'); if(!progression.length) empty(progressionTarget,'No progression snapshots or unlock events were recorded.'); else {{ const unlockedKinds=(kind)=>new Set(progression.filter(e=>{{const p=e.payload||{{}}; const status=String(p.status||'unlocked').toLowerCase(); return (p.item_kind===kind||p.catalog_kind===kind) && !['locked','failed','unknown'].includes(status);}}).map(e=>{{const p=e.payload||{{}};return p.item_id||p.catalog_id||JSON.stringify(p);}})).size; const statGrid=document.createElement('div');statGrid.className='grid';[['Map unlocks',unlockedKinds('map')],['Hero unlocks',unlockedKinds('hero')],['Progression events',progression.length]].forEach(([label,value])=>{{const card=document.createElement('div');card.className='card';const caption=document.createElement('span');caption.className='muted';caption.textContent=label;const number=document.createElement('span');number.className='value';number.textContent=value;card.append(caption,number);statGrid.append(card);}});progressionTarget.append(statGrid); const table=document.createElement('table'),head=document.createElement('tr'); ['Step','Event','Catalog / item','Status'].forEach(v=>{{const th=document.createElement('th');th.textContent=v;head.append(th);}});table.append(head);progression.forEach(e=>{{const p=e.payload||{{}},row=document.createElement('tr');row.append(cell(e.step_id),cell(e.kind),cell(p.catalog_kind||p.item_kind||p.item_id||''),cell(p.status||'unlocked'));table.append(row);}});progressionTarget.append(table); }}
  const matches=events.filter(e=>e.kind==='match.result'); const matchTarget=document.getElementById('matches-table'); if(!matches.length) empty(matchTarget,'No match results were recorded.'); else {{ const pvp=matches.filter(e=>e.payload?.match_kind==='pvp'), outcome=(e)=>String(e.payload?.outcome||'unknown').toLowerCase(), wins=pvp.filter(e=>['win','won','victory'].includes(outcome(e))).length, losses=pvp.filter(e=>['loss','lost','defeat'].includes(outcome(e))).length; const statGrid=document.createElement('div');statGrid.className='grid';[['Matches',matches.length],['Explicit PvP matches',pvp.length],['PvP wins',wins],['PvP losses',losses]].forEach(([label,value])=>{{const card=document.createElement('div');card.className='card';const caption=document.createElement('span');caption.className='muted';caption.textContent=label;const number=document.createElement('span');number.className='value';number.textContent=value;card.append(caption,number);statGrid.append(card);}});matchTarget.append(statGrid); const table=document.createElement('table'),head=document.createElement('tr'); ['Step','Type','Outcome','Turns','Trophy Δ'].forEach(v=>{{const th=document.createElement('th');th.textContent=v;head.append(th);}});table.append(head);matches.forEach(e=>{{const p=e.payload||{{}},row=document.createElement('tr');row.append(cell(e.step_id),cell(p.match_kind||'unknown'),cell(p.outcome||'unknown'),cell(p.turns),cell(p.trophy_delta));table.append(row);}});matchTarget.append(table); }}
  const artifactTarget=document.getElementById('artifact-table'); if(!artifacts.length) empty(artifactTarget,'No artifacts were registered.'); else {{ const table=document.createElement('table'),head=document.createElement('tr'); ['Role','Media','File','SHA-256'].forEach(v=>{{const th=document.createElement('th');th.textContent=v;head.append(th);}});table.append(head);artifacts.forEach(a=>{{const row=document.createElement('tr'),link=document.createElement('a');link.textContent=a.path;link.href=a.href;link.rel='noopener';row.append(cell(a.role),cell(a.media_type));const file=document.createElement('td');file.append(link);if(a.media_type.startsWith('image/')){{const image=document.createElement('img');image.className='media-preview';image.loading='lazy';image.src=a.href;image.alt=a.path;file.append(image);}}row.append(file,cell(a.sha256));table.append(row);}});artifactTarget.append(table); }}
  document.querySelectorAll('[data-panel]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-panel]').forEach(item=>item.setAttribute('aria-selected',item===button?'true':'false'));document.querySelectorAll('.panel').forEach(panel=>panel.classList.toggle('active',panel.id===button.dataset.panel));}}));
}})();
</script>
</body>
</html>
"##,
        data = data_json
    )
}
