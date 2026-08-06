"""
demo_app.py  --  a browser interface to the Denial-of-Wallet detector
======================================================================
This is the artefact demonstration. It generates a month of serverless request
traffic at a chosen attack class and intensity, renders it as the 24 x 30
heat-map the detector consumes, and runs BOTH models on it live:

  * the convolutional detector (dow_cnn_locked.pt), with its Grad-CAM attribution
  * the logistic-regression control, fitted on startup from the same dataset
  * the one-parameter mean-brightness threshold, calibrated on startup

Nothing here is precomputed or faked. Every number on the page comes from a
forward pass performed when you press the button.

    pip install flask                 # only dependency not already pinned
    python demo_app.py
    # open http://127.0.0.1:5000

Options:
    --data   dataset used to fit the control and calibrate the threshold
    --model  checkpoint to load
    --port   default 5000
"""
import argparse, base64, io, warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from flask import Flask, request, jsonify, render_template_string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

from dow_data import CLASS_NAMES, GenParams, generate_dataset, normalize
from dow_model import DoWNetCNN

warnings.filterwarnings("ignore")
app = Flask(__name__)
STATE = {}


# ─────────────────────────────────────────────────────────── rendering helpers
def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def heatmap_png(img, cmap="viridis", title=None):
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.imshow(img, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("day of month", fontsize=8, color="#8B94A3")
    ax.set_ylabel("hour of day", fontsize=8, color="#8B94A3")
    ax.set_xticks([0, 9, 19, 29]); ax.set_xticklabels([1, 10, 20, 30], fontsize=7)
    ax.set_yticks([0, 6, 12, 18, 23]); ax.set_yticklabels([0, 6, 12, 18, 23], fontsize=7)
    ax.tick_params(colors="#8B94A3", length=2)
    for sp in ax.spines.values():
        sp.set_color("#2A3242")
    if title:
        ax.set_title(title, fontsize=9, color="#C8D0DA")
    return png(fig)


# ─────────────────────────────────────────────────────────────────── inference
def gradcam(model, x, target):
    from captum.attr import LayerGradCam
    a = LayerGradCam(model, model.block2).attribute(x.unsqueeze(0), target=int(target))
    a = torch.nn.functional.interpolate(a, size=(24, 30), mode="bilinear",
                                        align_corners=False)
    m = np.abs(a.squeeze().detach().numpy())
    return m / m.max() if m.max() > 0 else m


def classify(raw_img):
    """raw_img: (1,24,30) raw counts. Returns every model's verdict."""
    model, lr, thr = STATE["model"], STATE["lr"], STATE["thr"]
    norm = normalize(raw_img[None, ...])                 # (1,1,24,30)
    x = torch.tensor(norm)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, 1).squeeze().numpy()
    cnn_pred = int(probs.argmax())

    lr_probs = lr.predict_proba(norm.reshape(1, -1)).squeeze()
    lr_pred = int(lr_probs.argmax())

    brightness = float(norm.mean())
    flagged = brightness >= thr

    cam = gradcam(model, x.squeeze(0), cnn_pred)
    return {
        "cnn": {"pred": cnn_pred, "probs": [round(float(p), 4) for p in probs]},
        "lr":  {"pred": lr_pred,  "probs": [round(float(p), 4) for p in lr_probs]},
        "thr": {"brightness": round(brightness, 4), "threshold": round(thr, 4),
                "flagged": bool(flagged)},
        "cam": cam,
        "norm": norm.squeeze(),
    }


def make_sample(cls_idx, intensity_scale, seed):
    """Generate one month of the requested class at the requested leech scale."""
    want_tag = 0 if cls_idx == 0 else 1          # 0 normal, 1 leech, 2 flood
    if intensity_scale >= 0.999 and cls_idx != 0:
        want_tag = 2                              # full strength -> flood variant
    params = GenParams(leech_scale=max(intensity_scale, 1e-3))
    X, y, tag = generate_dataset(per_class=4, seed=seed, params=params)
    hits = np.where((y == cls_idx) & (tag == want_tag))[0]
    if len(hits) == 0:
        hits = np.where(y == cls_idx)[0]
    return X[hits[0]]


# ───────────────────────────────────────────────────────────────────── routes
@app.route("/")
def index():
    return render_template_string(PAGE, classes=CLASS_NAMES)


@app.route("/run", methods=["POST"])
def run():
    d = request.get_json(force=True)
    cls = int(d.get("cls", 1))
    scale = float(d.get("scale", 1.0))
    seed = int(np.random.randint(0, 10_000))

    raw = make_sample(cls, scale, seed)
    r = classify(raw)
    names = list(CLASS_NAMES)

    return jsonify({
        "seed": seed,
        "truth": names[cls],
        "scale": scale,
        "input_png": heatmap_png(r["norm"]),
        "cam_png": heatmap_png(r["cam"], cmap="inferno"),
        "cnn_pred": names[r["cnn"]["pred"]], "cnn_probs": r["cnn"]["probs"],
        "cnn_ok": names[r["cnn"]["pred"]] == names[cls],
        "lr_pred": names[r["lr"]["pred"]], "lr_probs": r["lr"]["probs"],
        "lr_ok": names[r["lr"]["pred"]] == names[cls],
        "brightness": r["thr"]["brightness"], "threshold": r["thr"]["threshold"],
        "flagged": r["thr"]["flagged"],
        "truth_is_attack": cls != 0,
        "names": names,
    })


# ─────────────────────────────────────────────────────────────────────── page
PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Denial-of-Wallet detector — live demo</title>
<style>
*{box-sizing:border-box} body{margin:0;background:#10141C;color:#E6EAF0;
 font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:34px 26px 60px}
h1{font-size:26px;margin:0 0 4px;font-weight:650}
.sub{color:#7F8A9C;margin:0 0 26px;font-size:14px}
.panel{background:#161B26;border-radius:10px;padding:20px 22px;margin-bottom:18px}
.controls{display:flex;gap:26px;align-items:flex-end;flex-wrap:wrap}
label{display:block;font-size:12px;color:#8B94A3;margin-bottom:7px;
 letter-spacing:.04em;text-transform:uppercase}
select,input[type=range]{background:#0C1017;color:#E6EAF0;border:1px solid #2A3242;
 border-radius:6px;padding:9px 11px;font-size:14px}
input[type=range]{width:250px;padding:0;accent-color:#17A2A5}
button{background:#17A2A5;color:#04191A;border:0;border-radius:6px;
 padding:11px 24px;font-size:14px;font-weight:650;cursor:pointer}
button:disabled{opacity:.45;cursor:default}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.imgcard{background:#0C1017;border-radius:8px;padding:14px}
.imgcard h3{margin:0 0 4px;font-size:13px;font-weight:600;color:#C8D0DA}
.imgcard p{margin:0 0 10px;font-size:12px;color:#6D7788}
.imgcard img{width:100%;display:block}
.verdicts{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}
.v{background:#0C1017;border-radius:8px;padding:16px 17px}
.v .who{font-size:11px;color:#8B94A3;text-transform:uppercase;letter-spacing:.05em}
.v .what{font-size:21px;font-weight:650;margin:7px 0 3px}
.v .note{font-size:12px;color:#6D7788;line-height:1.45}
.ok{color:#3DD6A0} .bad{color:#E8734A}
.bars{margin-top:11px}
.bar{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:11.5px}
.bar span:first-child{width:66px;color:#8B94A3}
.bar .track{flex:1;height:6px;background:#1C2331;border-radius:3px;overflow:hidden}
.bar .fill{height:100%;background:#17A2A5}
.bar span:last-child{width:44px;text-align:right;color:#8B94A3}
.meta{margin-top:16px;font-size:12px;color:#5D6675}
.hint{font-size:12.5px;color:#7F8A9C;margin-top:12px;line-height:1.6}
.hint b{color:#C8D0DA;font-weight:600}
</style></head><body><div class="wrap">

<h1>Denial-of-Wallet detector</h1>
<p class="sub"><p class="sub">Itunu Deborah Adedoyin — x24249700 — MSc Cloud Computing, National College of Ireland</p></p>

<div class="panel"><div class="controls">
  <div><label>Traffic class</label>
    <select id="cls">
      {% for c in classes %}<option value="{{loop.index0}}"{% if loop.index0==1 %} selected{% endif %}>{{c}}</option>{% endfor %}
    </select></div>
  <div><label>Attack intensity — <span id="sv">1.00</span></label>
    <input type="range" id="scale" min="0.1" max="1.0" step="0.1" value="1.0"></div>
  <button id="go">Generate &amp; classify</button>
</div>
<p class="hint">Drag the intensity down and re-run. Below roughly <b>0.3</b> the attack
becomes a low-rate leech and both trained models start reading it as legitimate
traffic — while the one-parameter brightness rule often still catches it.</p></div>

<div id="out" style="display:none">
  <div class="grid">
    <div class="imgcard"><h3>Input — what the detector sees</h3>
      <p>24 hours down, 30 days across, normalised invocation count per cell</p>
      <img id="inimg"></div>
    <div class="imgcard"><h3>Grad-CAM — where it looked</h3>
      <p>attribution for the predicted class, brighter = more influence</p>
      <img id="camimg"></div>
  </div>
  <div class="verdicts">
    <div class="v"><div class="who">Convolutional detector</div>
      <div class="what" id="cnnp">—</div>
      <div class="note" id="cnnn"></div><div class="bars" id="cnnb"></div></div>
    <div class="v"><div class="who">Logistic regression control</div>
      <div class="what" id="lrp">—</div>
      <div class="note" id="lrn"></div><div class="bars" id="lrb"></div></div>
    <div class="v"><div class="who">One-parameter threshold</div>
      <div class="what" id="thp">—</div>
      <div class="note" id="thn"></div></div>
  </div>
  <div class="meta" id="meta"></div>
</div>

<script>
const $=id=>document.getElementById(id);
$('scale').oninput=e=>$('sv').textContent=(+e.target.value).toFixed(2);
function bars(el,probs,names,pred){
  el.innerHTML=probs.map((p,i)=>`<div class="bar"><span>${names[i]}</span>
    <span class="track"><span class="fill" style="width:${(p*100).toFixed(1)}%;
    ${i===pred?'':'opacity:.35'}"></span></span>
    <span>${(p*100).toFixed(1)}%</span></div>`).join('');
}
$('go').onclick=async()=>{
  const b=$('go'); b.disabled=true; b.textContent='Running…';
  try{
    const r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cls:+$('cls').value,scale:+$('scale').value})});
    const d=await r.json();
    $('out').style.display='block';
    $('inimg').src=d.input_png; $('camimg').src=d.cam_png;

    $('cnnp').textContent=d.cnn_pred; $('cnnp').className='what '+(d.cnn_ok?'ok':'bad');
    $('cnnn').textContent=d.cnn_ok?'correct':'wrong — true class was '+d.truth;
    bars($('cnnb'),d.cnn_probs,d.names,d.names.indexOf(d.cnn_pred));

    $('lrp').textContent=d.lr_pred; $('lrp').className='what '+(d.lr_ok?'ok':'bad');
    $('lrn').textContent=d.lr_ok?'correct':'wrong — true class was '+d.truth;
    bars($('lrb'),d.lr_probs,d.names,d.names.indexOf(d.lr_pred));

    const right=d.flagged===d.truth_is_attack;
    $('thp').textContent=d.flagged?'attack':'legitimate';
    $('thp').className='what '+(right?'ok':'bad');
    $('thn').textContent='mean brightness '+d.brightness.toFixed(4)+
      ' against a fixed threshold of '+d.threshold.toFixed(4)+
      ' — calibrated once at full intensity and never refitted.';

    $('meta').textContent='true class: '+d.truth+'   ·   intensity scale: '+
      d.scale.toFixed(2)+'   ·   generator seed: '+d.seed;
  }catch(e){ alert('Request failed: '+e); }
  b.disabled=false; b.textContent='Generate & classify';
};
</script></div></body></html>"""


# ───────────────────────────────────────────────────────────────────── startup
def boot(args):
    print("loading model …")
    model = DoWNetCNN(n_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    print("fitting logistic-regression control …")
    d = np.load(args.data, allow_pickle=True)
    Xn = normalize(d["X"]).reshape(len(d["y"]), -1)
    lr = LogisticRegression(max_iter=2000).fit(Xn, d["y"])

    print("calibrating one-parameter threshold …")
    X1, y1, t1 = generate_dataset(per_class=300, seed=0, params=GenParams())
    s = normalize(X1).mean(axis=(1, 2, 3))
    keep = (t1 == 0) | (t1 == 1)
    fpr, tpr, thr = roc_curve((t1[keep] == 1).astype(int), s[keep])
    STATE.update(model=model, lr=lr, thr=float(thr[np.argmax(tpr - fpr)]))
    print(f"ready — threshold {STATE['thr']:.4f}\n"
          f"listening on 0.0.0.0:{args.port} — use Preview > Preview Running Application\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lramp.npz")
    ap.add_argument("--model", default="dow_cnn_locked.pt")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    boot(a)
    app.run(host="0.0.0.0", port=a.port, debug=False)
