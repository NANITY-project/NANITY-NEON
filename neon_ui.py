#!/usr/bin/env python3
"""neon_ui.py - one-file local UI for the NEON idle loop (stdlib only).

Put it in the repo root, then:
    python3 neon_ui.py --model models/SmolLM2-1.7B-nanity-Q4_0.gguf
    python3 neon_ui.py --model https://huggingface.co/<user>/<repo>/resolve/main/<file>.gguf
    python3 neon_ui.py --model some-plain.gguf      # auto-converted with nanity_convert.py (needs numpy)
    python3 neon_ui.py --selftest                   # tests the server against a fake neon, no model needed

It builds ./neon if missing/stale (-DNANITY_ENABLE_IDLE_LOOP; fetches nectar_cot/splice from Project-Infinite if absent), fetches/converts the model,
runs `neon --idle-loop --idle-cot`, and bridges its stdin/stdout to a browser chat page.
"""
import argparse, codecs, http.client, json, os, re, subprocess, sys, tempfile, threading, time, urllib.request, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


class Hub:
    def __init__(s):
        s.ev, s.base, s.cv = [], 0, threading.Condition()
        s.proc, s.gen = None, 0

    def push(s, **e):
        with s.cv:
            s.ev.append(e)
            if len(s.ev) > 20000:
                s.ev, s.base = s.ev[10000:], s.base + 10000
            s.cv.notify_all()

    def status(s, state, msg=""):
        s.push(t="status", state=state, msg=msg or state)


hub = Hub()


def run_logged(cmd, tag):
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-15:]
        hub.push(t="err", d=f"[{tag}] {line.rstrip()}")
    return p.wait(), "\n".join(tail)


INFINITE = "https://raw.githubusercontent.com/NANITY-project/Project-Infinite/main/"


def ensure_idle_files():
    """The idle-loop headers live in the Project-Infinite repo; NEON's own copy of nectar_splice.hpp is older
    and nectar_cot.hpp is absent. Fetch the two that the --idle-cot build needs if cot is missing."""
    d = ROOT / "runtime" / "idle_loop"
    if (d / "nectar_cot.hpp").exists():
        return
    hub.status("building", "fetching idle-loop headers from Project-Infinite")
    for name in ("nectar_cot.hpp", "nectar_splice.hpp"):
        if name == "nectar_splice.hpp" and (d / name).exists():
            (d / (name + ".orig")).write_bytes((d / name).read_bytes())
        try:
            with urllib.request.urlopen(INFINITE + name, timeout=30) as r:
                (d / name).write_bytes(r.read())
        except Exception as e:
            raise RuntimeError(f"could not fetch {name} from Project-Infinite ({e}); copy it into runtime/idle_loop/ by hand")


def ensure_build(a):
    if a.neon_bin:
        return Path(a.neon_bin)
    exe = ROOT / "neon"
    src = [p.stat().st_mtime for p in (ROOT / "runtime").rglob("*") if p.is_file()]
    if exe.exists() and src and exe.stat().st_mtime >= max(src):
        return exe
    ensure_idle_files()
    cpu = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""
    flags = [f for n, f in (("avx2", "-mavx2"), ("fma", "-mfma"), ("f16c", "-mf16c")) if re.search(rf"\b{n}\b", cpu)]
    hub.status("building", "building NEON (first run takes a while)")
    cmd = [os.environ.get("CXX", "g++"), "-std=c++20", "-O2", "-pthread", "-DNANITY_ENABLE_IDLE_LOOP", *flags,
           "runtime/NEON-3.cpp", "-o", "neon.tmp"]
    rc, tail = run_logged(cmd, "build")
    if rc:
        raise RuntimeError("build failed:\n" + tail)
    os.replace(ROOT / "neon.tmp", exe)
    return exe


def fetch(url, dest):
    dest.parent.mkdir(exist_ok=True)
    part, last = dest.with_suffix(dest.suffix + ".part"), 0
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        total, got = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 1:
                last = time.time()
                hub.status("downloading", f"downloading model {got >> 20}/{total >> 20} MB")
    os.replace(part, dest)


def ensure_model(a, exe):
    m = a.model
    if not m:
        c = sorted((ROOT / "models").glob("*.gguf")) if (ROOT / "models").exists() else []
        if not c:
            raise RuntimeError("no model given: use --model <path-or-url> (or put a .gguf in ./models)")
        m = str(c[0])
    if m.startswith("http"):
        dest = ROOT / "models" / m.split("?")[0].rsplit("/", 1)[1]
        if not dest.exists():
            fetch(m, dest)
        m = str(dest)
    m = Path(m)
    if not m.exists():
        raise RuntimeError(f"model not found: {m}")
    hub.status("probing", "checking model")
    if subprocess.run([str(exe), "--model", str(m), "--probe"], cwd=ROOT, capture_output=True).returncode == 0:
        return m
    out = m.with_name(m.stem + "-nanity.gguf")
    if not out.exists():
        hub.status("converting", "converting to NANITY format (Q4_0)")
        rc, tail = run_logged([sys.executable, "conversion/nanity_convert.py", str(m), str(out), "--quant", "Q4_0"], "convert")
        if rc:
            raise RuntimeError("conversion failed (try --detect-only on it):\n" + tail)
    return out


def pump_out(p, gen, logf):
    dec = codecs.getincrementaldecoder("utf-8")("replace")
    while True:
        b = os.read(p.stdout.fileno(), 4096)
        if not b:
            break
        t = dec.decode(b)
        if t:
            logf.write(t)
            logf.flush()
            hub.push(t="out", d=t)


def pump_err(p):
    for line in p.stderr:
        hub.push(t="err", d=line.decode("utf-8", "replace").rstrip())


def pipeline(a, gen):
    try:
        exe = ensure_build(a)
        model = ensure_model(a, exe)
        cmd = [str(exe), "--model", str(model), "--idle-loop", "--idle-cot"]
        if a.threads:
            cmd += ["--threads", str(a.threads)]
        cmd += a.extra
        hub.status("starting", "loading model")
        p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        hub.proc = p
        (ROOT / "logs").mkdir(exist_ok=True)
        logf = open(ROOT / "logs" / time.strftime("neon_ui_%Y%m%d_%H%M%S.txt"), "w", encoding="utf-8")
        threading.Thread(target=pump_err, args=(p,), daemon=True).start()
        hub.status("running", f"running: {Path(model).name}")
        pump_out(p, gen, logf)
        rc = p.wait()
        if gen == hub.gen:
            hub.status("stopped", f"NEON exited (code {rc})")
    except Exception as e:
        hub.push(t="err", d=str(e))
        hub.status("error", str(e).splitlines()[0])


def start(a):
    hub.gen += 1
    threading.Thread(target=pipeline, args=(a, hub.gen), daemon=True).start()


def kill():
    p = hub.proc
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()


def make_handler(a):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def reply(self, code, body=b"", ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self.reply(200, PAGE.encode(), "text/html; charset=utf-8")
            if path != "/events":
                return self.reply(404)
            i = int(self.headers.get("Last-Event-ID", -1)) + 1
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with hub.cv:
                        hub.cv.wait_for(lambda: i < hub.base + len(hub.ev), timeout=15)
                        i = max(i, hub.base)
                        batch = hub.ev[i - hub.base:]
                    if not batch:
                        self.wfile.write(b": ka\n\n")
                    for n, e in enumerate(batch):
                        self.wfile.write(f"id: {i + n}\ndata: {json.dumps(e)}\n\n".encode())
                    i += len(batch)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            p = hub.proc
            if path == "/send":
                if not p or p.poll() is not None:
                    return self.reply(409, b"not running")
                p.stdin.write((" ".join(str(data.get("text", "")).split()) + "\n").encode())
                p.stdin.flush()
                return self.reply(200, b"ok")
            if path == "/restart":
                hub.gen += 1
                kill()
                hub.push(t="reset")
                return (start(a), self.reply(200, b"ok"))[1]
            if path == "/stop":
                hub.gen += 1
                kill()
                hub.status("stopped", "stopped")
                return self.reply(200, b"ok")
            self.reply(404)
    return H


def serve(a):
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(a))
    start(a)
    url = f"http://{a.host}:{srv.server_port}/"
    print("NEON UI:", url)
    if a.open:
        webbrowser.open(url)
    return srv, url


FAKE = r'''#!/usr/bin/env python3
import sys, time
if "--probe" in sys.argv: sys.exit(0)
def out(s): sys.stdout.write(s); sys.stdout.flush()
out("[fake-build]\n<think>")
import threading
def reader():
    for line in sys.stdin:
        out("\n[USER] " + line.strip() + "\n<answer>echo " + line.strip() + "</answer>\n<think>")
threading.Thread(target=reader, daemon=True).start()
while True:
    out("thinking. "); time.sleep(0.05)
'''


def selftest():
    d = Path(tempfile.mkdtemp())
    fake = d / "fakeneon"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    (d / "m.gguf").write_bytes(b"x")
    a = argparse.Namespace(model=str(d / "m.gguf"), neon_bin=str(fake), threads=None, extra=[], host="127.0.0.1", port=0, open=False)
    srv, _ = serve(a)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/events")
    r = conn.getresponse()
    seen, sent, deadline = "", False, time.time() + 10
    while time.time() < deadline and "echo ping" not in seen:
        line = r.fp.readline().decode()
        if line.startswith("data: "):
            e = json.loads(line[6:])
            seen += e.get("d", "") if e["t"] == "out" else ""
            if e["t"] == "status" and e["state"] == "running" and not sent:
                c2 = http.client.HTTPConnection("127.0.0.1", port)
                c2.request("POST", "/send", json.dumps({"text": "ping"}))
                assert c2.getresponse().status == 200
                sent = True
    kill()
    assert "[fake-build]" in seen and "[USER] ping" in seen and "<answer>echo ping</answer>" in seen, seen[-300:]
    page_ok = http.client.HTTPConnection("127.0.0.1", port)
    page_ok.request("GET", "/")
    assert b"NEON" in page_ok.getresponse().read()
    print("selftest OK: server spawned the process, streamed stdout, relayed stdin, served the page")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>NEON idle loop</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fff;--fg:#1c1c1e;--mut:#8a8a8e;--card:#f2f2f4;--me:#2b6cff;--line:#dcdce0}
@media(prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e8ea;--mut:#8a8a90;--card:#232327;--me:#3b78ff;--line:#34343a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;display:flex;flex-direction:column;height:100vh}
header{display:flex;gap:10px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);flex-wrap:wrap;font-size:13px}
header b{letter-spacing:.08em}.sp{flex:1}button{font:inherit;padding:3px 10px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;cursor:pointer}
#chat{flex:1;overflow:auto;padding:14px;display:flex;flex-direction:column;gap:8px}
.bub{max-width:80%;padding:8px 12px;border-radius:14px;white-space:pre-wrap;background:var(--card)}.me{align-self:flex-end;background:var(--me);color:#fff}
.ask{border:1px solid var(--me)}.think,.next{color:var(--mut);font-style:italic;white-space:pre-wrap;font-size:13px;border-left:2px solid var(--line);padding-left:8px}
.next{font-size:12px}.sys{color:var(--mut);font-size:12px;white-space:pre-wrap}body.hide .think,body.hide .next{display:none}
#log{display:none;max-height:20vh;overflow:auto;margin:0;padding:6px 12px;font:11px monospace;background:var(--card);color:var(--mut);border-top:1px solid var(--line)}
footer{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--line)}input{flex:1;font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
</style></head><body>
<header><b>NEON</b><span id="state">connecting</span><span id="stats" style="color:var(--mut)"></span><span class="sp"></span>
<label><input type="checkbox" id="th" checked> show thinking</label><button id="lg">log</button><button id="sv">save</button><button id="rs">restart</button><button id="sp">stop</button></header>
<div id="chat"></div><pre id="log"></pre>
<footer><input id="in" placeholder="Talk to it - spliced in at the end of the current sentence" autofocus><button id="go">send</button></footer>
<script>
const $=id=>document.getElementById(id),chat=$('chat');
/*PARSER*/
let buf='',mode=null,cur=null,ask=false,pend=0;
const TAG=/<\/?(think|answer|next)>|\[USER\] ?|\[ASK\] ?/;
function add(t){if(!t||(!mode&&!t.trim()))return;const m=mode||'sys';
 if(!cur||cur.dataset.m!==m){cur=document.createElement('div');cur.dataset.m=m;
  cur.className=m==='answer'?'bub'+(ask?' ask':''):m==='user'?'bub me':m;chat.appendChild(cur);
  while(chat.children.length>1500)chat.removeChild(chat.firstChild)}
 cur.textContent+=t;if(chat.scrollHeight-chat.scrollTop-chat.clientHeight<200)chat.scrollTop=chat.scrollHeight}
function feed(s){buf+=s;for(;;){const m=TAG.exec(buf);
 if(!m){const k=Math.max(buf.lastIndexOf('<'),buf.lastIndexOf('['));
  if(k>=0&&buf.length-k<12){add(buf.slice(0,k));buf=buf.slice(k)}else{add(buf);buf=''}return}
 add(buf.slice(0,m.index));buf=buf.slice(m.index+m[0].length);const t=m[0];
 if(t[0]==='<'){if(t[1]==='/'){if(m[1]==='answer')ask=false;mode=null}else mode=m[1]}
 else if(t.startsWith('[USER]')){mode='user';ask=false;if(pend>0)pend--;queued()}else ask=true;
 cur=null}}
/*END*/
function queued(){$('stats').dataset.q=pend?' | queued: '+pend:'';stat()}
let chars=0,rate=0,cycles=0,shifts=0;
function stat(){$('stats').textContent=(cycles?'cycle '+cycles+' | ':'')+rate+' chars/s | shift/compress log lines: '+shifts+($('stats').dataset.q||'')}
setInterval(()=>{rate=chars;chars=0;stat()},1000);
const es=new EventSource('/events');
es.onmessage=e=>{const d=JSON.parse(e.data);
 if(d.t==='out'){chars+=d.d.length;feed(d.d)}
 else if(d.t==='status'){$('state').textContent=d.msg}
 else if(d.t==='reset'){chat.innerHTML='';buf='';mode=cur=null;pend=0;cycles=shifts=0}
 else if(d.t==='err'){const l=$('log');l.textContent+=d.d+'\n';l.scrollTop=l.scrollHeight;
  const c=/cycle (\d+)/.exec(d.d);if(c)cycles=+c[1];if(/zram|shift|summar|compress/i.test(d.d))shifts++}};
async function send(){const v=$('in').value.trim();if(!v)return;$('in').value='';
 const r=await fetch('/send',{method:'POST',body:JSON.stringify({text:v})});
 if(r.ok){pend++;queued()}else $('state').textContent='not running'}
$('go').onclick=send;$('in').onkeydown=e=>{if(e.key==='Enter')send()};
$('th').onchange=e=>document.body.classList.toggle('hide',!e.target.checked);
$('lg').onclick=()=>{const l=$('log');l.style.display=l.style.display==='block'?'none':'block'};
$('rs').onclick=()=>fetch('/restart',{method:'POST',body:'{}'});$('sp').onclick=()=>fetch('/stop',{method:'POST',body:'{}'});
$('sv').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([chat.innerText]));a.download='neon_transcript.txt';a.click()};
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="path or http(s) URL of a GGUF (plain GGUFs are converted automatically)")
    ap.add_argument("--neon-bin", help="use this neon binary instead of building ./neon")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", dest="open", action="store_false")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra neon flags, after --")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    srv, _ = serve(a)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        kill()


if __name__ == "__main__":
    main()
