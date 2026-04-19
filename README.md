# SumANN — Deployment Bundle

Self-contained package for running the trained SumANN model **anywhere**
an ONNX runtime is available, including directly in the web browser.

## Contents

| File | Purpose |
|---|---|
| `model.onnx` | Trained neural network exported to ONNX (opset 17, dynamic batch axis). |
| `scaler_params.json` | StandardScaler mean and scale, plus input/output column names and units. Required for preprocessing. |
| `index.html` | Drop-in demo that runs the model in a web browser via `onnxruntime-web`. |
| `MODEL_CARD.md` | Model documentation: intended use, performance metrics, caveats, citation. |

## Sharing via GitHub Pages (recommended)

The demo is pure HTML + JS + ONNX — no server, no build step. GitHub Pages
hosts it for free directly from a public repository.

**One-time setup:**

1. Create a new public repository on GitHub (e.g. `sumann-demo`).
2. Copy the four runtime files — `model.onnx`, `scaler_params.json`,
   `index.html`, `README.md` — into the repository root.
3. Push to the `main` branch.
4. In the repository's **Settings → Pages**, set:
   - *Source*: Deploy from a branch
   - *Branch*: `main` / `(root)`
5. Wait ~1 minute for GitHub to build and publish.

Your collaborator opens `https://<your-username>.github.io/sumann-demo/`
and the model runs live in their browser. No installation on their side.
No account needed to view.

**To share updates**, just commit new versions of the four files and push.
GitHub Pages republishes automatically within a minute.

**Note:** the repository must be public on the free plan. If the model
is commercially sensitive, see the "Private hosting" section below.

## Running locally

```bash
cd outputs/deploy
python3 -m http.server 8000
# open http://localhost:8000 in any modern browser
```

Opening `index.html` directly via `file://` is blocked by the browser's
same-origin rules for `fetch`, so the local HTTP server step is
required.

## Running in Python

```python
import json
import numpy as np
import onnxruntime as ort

with open("scaler_params.json") as f:
    sp = json.load(f)

sess = ort.InferenceSession("model.onnx")

# Example: one sample in the ORIGINAL (unscaled) units
raw = np.array([[1.2, 3.4, 5.6]], dtype=np.float32)
scaled = ((raw - np.array(sp["mean"])) / np.array(sp["scale"])).astype(np.float32)

pred = sess.run(None, {"input": scaled})[0]
print(dict(zip(sp["output_cols"], pred[0])))
```

## Private hosting alternatives

If the model should not be public, the following free options keep the
source private while still giving collaborators a shareable URL:

- **Cloudflare Pages** — private source on a private GitHub repo, public
  site URL, no visitor account required. Free tier is generous.
- **Netlify** — same pattern; free for hobby use.
- **Hugging Face Spaces** — free, ML-focused, native ONNX support, but
  the deployed page is public by default unless you pay for a private
  Space.

In all three, upload the same four files and point the host at the
directory containing `index.html`.

## Running in any ONNX-compatible runtime

The `.onnx` file is a standard opset-17 graph with a single input named
`input` (shape `[batch, n_features]`, dtype `float32`) and a single
output named `output` (shape `[batch, n_targets]`, dtype `float32`).
Compatible with:

- `onnxruntime` — Python, C, C++, C#, Java.
- `onnxruntime-web` — JavaScript (WASM backend).
- `onnxruntime-node` — Node.js.
- `onnxruntime-mobile` — iOS, Android.
- TensorRT, OpenVINO, CoreML via standard ONNX conversion tools.

**Preprocessing is the caller's responsibility.** Apply
`(x − mean) / scale` per feature before inference. The model does not
scale its output — predictions are returned in the original target
units.

---

© Siddharth Suman, Ph.D. &nbsp;·&nbsp; www.siddharthsuman.com
