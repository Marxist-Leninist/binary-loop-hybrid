"""
Multi-run dashboard for the fully-binary variant runs.

Stdlib-only HTTP server. Reads runs/*/state.json (live) and results.json
(final comparison, written by eval2.py) from the directory it lives in.
Auto-refreshes. Usage: python3 dashboard2.py [port]
"""

import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

LABELS = {
    "tf-base": "A. Binary looped transformer (unlikelihood OFF)",
    "tf-ul": "A'. Binary looped transformer (unlikelihood ON)",
    "mamba": "B. Binary looped Mamba",
    "hybrid": "C. Fully binary hybrid",
}


def load_runs():
    runs = []
    if os.path.isdir(RUNS):
        for name in sorted(os.listdir(RUNS)):
            p = os.path.join(RUNS, name, "state.json")
            meta_p = os.path.join(RUNS, name, "run_meta.json")
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        st = json.load(f)
                except Exception:  # noqa: BLE001
                    continue
                meta = {}
                if os.path.exists(meta_p):
                    try:
                        with open(meta_p) as f:
                            meta = json.load(f)
                    except Exception:  # noqa: BLE001
                        pass
                st["_meta"] = meta
                st["_done"] = os.path.exists(os.path.join(RUNS, name, "DONE"))
                runs.append(st)
    order = {"tf-base": 0, "tf-ul": 1, "mamba": 2, "hybrid": 3}
    runs.sort(key=lambda r: order.get(r["run"], 9))
    return runs


def load_results():
    p = os.path.join(HERE, "results.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            return None
    return None


def page():
    runs = load_runs()
    cards = []
    keyed = {r["run"]: r for r in runs}
    for run_id, label in LABELS.items():
        label = html.escape(label)
        r = keyed.get(run_id)
        if r is None:
            cards.append(f'<article class="card queued"><h2>{label}</h2><span class="pill">Queued</span><p>No training metrics yet.</p></article>')
            continue
        m = r.get("_meta", {})
        done = bool(r.get("_done"))
        target = m.get("steps")
        step = int(r.get("step", 0))
        out_step = f"{step:,}" + (f" / {target:,}" if isinstance(target,int) and target>0 else "")
        prog = f'<progress value="{min(step,target)}" max="{target}" aria-label="Training steps"></progress>' if isinstance(target,int) and target>0 else ""
        status = "Finished" if done else "Training"
        sample = r.get("sample") or ""
        repeat = r.get("sample_repeat4")
        repeat_label = f"{repeat:,} repeated 4-grams" if isinstance(repeat,int) else "Not measured yet"
        rows = [
            ("Recent CE", r.get("loss", "—")),
            ("Smoothed CE", r.get("ema_loss", "—")),
            ("Throughput", f"{r.get('tok_s',0):,.0f} tok/s"),
            ("Parameters", f"{r.get('params',0):,}"),
            ("Binary latents", f"{r.get('binary_latent_params',0):,}"),
            ("Peak GPU memory", f"{r.get('gpu_mem_mb',0):,.1f} MiB"),
        ]
        metrics = "".join(f"<div><dt>{name}</dt><dd>{html.escape(str(value))}</dd></div>" for name,value in rows)
        detail = (f'<details data-run="{run_id}"><summary>Raw sample <span>{repeat_label}</span></summary>'
                  f'<pre class="sample">{html.escape(sample)}</pre></details>' if sample else
                  '<p class="muted">Waiting for the first generated sample.</p>')
        cards.append(f"""
        <article class="card">
          <h2>{label}</h2>
          <div class="progressline"><span class="pill {('done' if done else 'active')}">{status}</span><strong>{out_step} steps</strong></div>
          {prog}
          <dl class="metrics">{metrics}</dl>
          {detail}
        </article>""")
    results = load_results()
    table = ""
    if results:
        rows = []
        for r in results:
            s = r["samples"]["Once upon a time"]
            rows.append(
                f"<tr><td>{html.escape(LABELS.get(r['run'], r['run']))}</td>"
                f"<td>{r['params']:,}</td><td>{r['final_loss']:.4f}</td>"
                f"<td>{r['final_ema_loss']:.4f}</td>"
                f"<td>{r['kv_bytes_per_token']} B/tok</td>"
                f"<td>{r['ssm_state_bytes']} B const</td>"
                f"<td>{s['repeat4']} / {s['n_grams']}</td></tr>")
        table = f"""<h2>Final comparison (matched steps, raw greedy)</h2>
        <table class="cmp"><tr><th>run</th><th>params</th><th>final CE</th>
        <th>EMA CE</th><th>KV growth</th><th>SSM state</th>
        <th>repeat-4grams</th></tr>{''.join(rows)}</table>"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Fully-Binary Variants · Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{{color-scheme:dark;font:15px/1.5 system-ui,sans-serif;background:#101620;color:#eaf0f7}}
*,*::before,*::after{{box-sizing:border-box}}
body{{margin:0}}
main{{max-width:1220px;margin:auto;padding:clamp(12px,2.5vw,28px)}}
header{{margin:0 0 18px}}
h1{{font-size:clamp(22px,4vw,34px);line-height:1.15;margin:0 0 4px}}
header p,.muted{{color:#adbed0;margin:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:12px}}
.card{{min-width:0;background:#192331;border:1px solid #324255;border-radius:12px;padding:16px}}
.card h2{{font-size:15px;margin:0 0 8px;overflow-wrap:anywhere}}
.queued p{{margin:8px 0 0;color:#adbed0}}
.progressline{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;font-variant-numeric:tabular-nums}}
.pill{{background:#26384c;color:#c8d6e5;border-radius:999px;padding:2px 8px;font-size:12px}}
.pill.active{{background:#123641;color:#8be6f6}}.pill.done{{background:#16392c;color:#a2eab8}}
progress{{width:100%;height:12px;accent-color:#70d9eb;margin:6px 0}}
.metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:6px 0 12px}}
.metrics div{{min-width:0;padding:7px 8px;border:1px solid #314154;border-radius:7px}}
dt{{color:#abc1d4;font-size:12px}}dd{{font-size:clamp(13px,2.2vw,16px);font-weight:650;overflow-wrap:anywhere;margin:0;font-variant-numeric:tabular-nums}}
details{{border-top:1px solid #3d4d61}}
summary{{color:#dbe8f5;cursor:pointer;min-height:40px;padding:10px 0;overflow-wrap:anywhere}}
summary span{{color:#ffc990;font-size:12px;padding-left:5px}}
.sample{{margin:0;background:#101923;padding:12px;border-radius:7px;color:#cbdcf0;max-height:15em;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}}
[tabindex]:focus-visible,summary:focus-visible{{outline:2px solid #7bdded;outline-offset:2px}}
.results{{overflow:auto;margin-top:18px}}.results:empty{{display:none}}
table.cmp{{border-collapse:collapse;font-size:13px}}table.cmp td,table.cmp th{{border:1px solid #3d4d61;text-align:left;padding:6px 11px;white-space:nowrap}}
@media(max-width:430px){{.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body><main>
<header><h1>Fully-Binary Variants</h1><p>Matched training runs · GTX 1660 SUPER · updates every 10 seconds</p>
<p id="refreshStatus" aria-live="polite"></p></header>
<div id="liveRuns" class="grid">{''.join(cards)}</div>
<section id="finalResults" class="results" tabindex="0" aria-label="Final comparison">{table}</section>
</main>
<script>
(() => {{
 const main = document.getElementById('liveRuns'), results = document.getElementById('finalResults');
 const status = document.getElementById('refreshStatus');
 async function refresh() {{
  try {{
   const response = await fetch(location.pathname, {{cache:'no-store'}});
   if (!response.ok) throw Error('Refresh failed');
   const next = new DOMParser().parseFromString(await response.text(),'text/html');
   const grid = next.getElementById('liveRuns'), table = next.getElementById('finalResults');
   if (!grid || !table) throw Error('Dashboard unavailable');
   const active = document.activeElement?.closest('details')?.dataset.run;
   const open = new Map([...main.querySelectorAll('details[open]')].map(d=>[d.dataset.run,d.querySelector('pre').scrollTop]));
   main.replaceChildren(...grid.childNodes);
   if (results.innerHTML !== table.innerHTML) results.replaceChildren(...table.childNodes);
   main.querySelectorAll('details').forEach(d=>{{
    if(open.has(d.dataset.run)) {{ d.open=true; d.querySelector('pre').scrollTop=open.get(d.dataset.run); }}
    if(active===d.dataset.run) d.querySelector('summary').focus({{preventScroll:true}});
   }});
   if(status.textContent)status.textContent='';
  }} catch {{ status.textContent='Live update interrupted. Showing the last loaded metrics.'; }}
  finally {{ setTimeout(refresh,10000); }}
 }}
 setTimeout(refresh,10000);
}})();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/state"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            runs = load_runs()
            for r in runs:
                r.pop("_meta", None)
            self.wfile.write(json.dumps(runs).encode())
            return
        body = page().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
