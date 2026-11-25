"""
Lightweight web dashboard for the Binary Looped Hybrid trainer.

Stdlib-only HTTP server. Serves:
  /         auto-refreshing HTML monitor (loss curve + live stats)
  /state    current state.json
  /history  history.jsonl (for the curve)
Run: python3 dashboard.py [port]   (default 8080)
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
HISTORY = os.path.join(HERE, "history.jsonl")

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Binary Looped Hybrid - LIVE TRAINING</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{background:#101418;color:#e8eef4;font-family:Consolas,Menlo,monospace;
      margin:16px}
 h1{color:#ffb000;font-size:17px;margin:0 0 4px}
 .muted{color:#8fa3b8;font-size:12px}
 .grid{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
 .cell{background:#171d24;padding:8px 12px;min-width:104px}
 .cell .lab{color:#8fa3b8;font-size:10px}
 .cell .val{font-size:19px;font-weight:bold}
 canvas{background:#171d24;width:100%;height:300px}
 #sample{background:#171d24;padding:10px;white-space:pre-wrap;font-size:13px}
</style></head><body>
<h1>BINARY LOOPED (INTERNAL-REASONING) TRANSFORMER - MAMBA HYBRID</h1>
<div class="muted" id="cfg">waiting for trainer...</div>
<div class="grid" id="stats"></div>
<canvas id="cv" width="1100" height="300"></canvas>
<div class="muted" style="margin-top:10px">LATEST GREEDY SAMPLE (prompt: 'The ')</div>
<div id="sample">(none yet)</div>
<div class="muted" id="foot" style="margin-top:8px"></div>
<script>
const FIELDS=[["step","STEP"],["loss","LOSS"],["ema_loss","EMA LOSS"],
 ["tok_s","TOKENS/S"],["lr","LR"],["gpu_mem_mb","GPU MEM MB"],
 ["elapsed_s","ELAPSED S"]];
let hist=[];
function draw(){
  const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
  if(hist.length<2){ctx.fillStyle='#8fa3b8';ctx.font='14px monospace';
    ctx.fillText('collecting loss points...',20,30);return;}
  const pts=hist.slice(-1500);
  const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),es=pts.map(p=>p[2]);
  const x0=Math.min(...xs),x1=Math.max(...xs);
  let y1=Math.max(...ys,...es)*1.05,y0=Math.min(...ys,...es)*0.95;
  const L=48,R=10,T=10,B=22;
  const X=x=>L+(x-x0)/Math.max(1,x1-x0)*(W-L-R);
  const Y=y=>T+(y1-y)/(y1-y0)*(H-T-B);
  ctx.strokeStyle='#232b34';ctx.fillStyle='#8fa3b8';ctx.font='11px monospace';
  for(const f of [0,0.5,1]){const yv=y0+f*(y1-y0);
    ctx.beginPath();ctx.moveTo(L,Y(yv));ctx.lineTo(W-R,Y(yv));ctx.stroke();
    ctx.fillText(yv.toFixed(2),4,Y(yv)+4);}
  ctx.fillText('step '+x0,L,H-6);
  const lbl='step '+x1;ctx.fillText(lbl,W-R-ctx.measureText(lbl).width,H-6);
  ctx.strokeStyle='#54677c';ctx.lineWidth=1;ctx.beginPath();
  pts.forEach((p,i)=>{const px=X(p[0]),py=Y(p[1]);
    i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();
  ctx.strokeStyle='#39d0c4';ctx.lineWidth=2;ctx.beginPath();
  pts.forEach((p,i)=>{const px=X(p[0]),py=Y(p[2]);
    i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();
  ctx.fillStyle='#39d0c4';ctx.fillText('EMA loss',W-R-70,T+12);
}
async function tick(){
  try{
    const s=await (await fetch('/state')).json();
    document.getElementById('cfg').textContent=
      'params '+s.params.toLocaleString()+' | binary-latent params '+
      s.binary_latent_params.toLocaleString()+' | reasoning loops R='+
      s.loops+' per looped block | d_model '+s.d_model+' | batch '+
      s.batch_size+' x seq '+s.seq_len+' | data: '+s.dataset+
      ' (HF streaming) | device '+s.device+' | trainer pid '+s.pid+
      ' | RTX 2080 Ti (Vast 61918457)';
    document.getElementById('stats').innerHTML=FIELDS.map(([k,lab])=>{
      let v=s[k];if(k==='lr')v=v.toExponential(2);
      if(k==='elapsed_s')v=Math.floor(v);
      return '<div class="cell"><div class="lab">'+lab+
             '</div><div class="val">'+v+'</div></div>';}).join('');
    document.getElementById('sample').textContent=s.sample||'(none yet)';
    document.getElementById('foot').textContent=
      'auto-refreshing every 2s - state step '+s.step;
    if(!hist.length||hist[hist.length-1][0]!==s.step){
      hist.push([s.step,s.loss,s.ema_loss]);draw();}
  }catch(e){
    document.getElementById('foot').textContent='waiting for state... '+e;
  }
}
async function seed(){
  try{
    const txt=await (await fetch('/history')).text();
    hist=txt.trim().split('\\n').filter(Boolean).map(l=>{
      const s=JSON.parse(l);return [s.step,s.loss,s.ema_loss];});
  }catch(e){}
  draw();
}
seed();tick();setInterval(tick,2000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        elif self.path == "/state":
            try:
                with open(STATE, "rb") as f:
                    self._send(f.read(), "application/json")
            except FileNotFoundError:
                self._send("{}", "application/json")
        elif self.path == "/history":
            try:
                with open(HISTORY, "rb") as f:
                    self._send(f.read(), "text/plain; charset=utf-8")
            except FileNotFoundError:
                self._send("", "text/plain; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
