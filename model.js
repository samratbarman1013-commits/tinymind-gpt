/* =========================================================================
   TinyMind GPT — pure-JavaScript GPT inference engine
   Reimplements the PyTorch model's forward pass with KV-caching (plus a
   sliding window at the context boundary) so generation can run
   indefinitely. 100% client-side, no server, no API.

   The core engine is module-exportable so it can be unit-tested in Node
   against the PyTorch reference before deployment.
   ========================================================================= */
(function (root) {
  "use strict";

  // ---- model constants (must match training config) ----
  const D = 192, L = 4, H = 6, HD = 32, FF = 768, V = 70, MAXSEQ = 128;
  const HEAD_SCALE = 1 / Math.sqrt(HD);
  const LN_EPS = 1e-5;

  // ---- base64 -> Int8 -> Float32 (dequantize int8 per-tensor quantization) ----
  function decodeWeights(w) {
    const out = {};
    for (const name in w) {
      const e = w[name];
      const bin = atob(e.data);
      const u8 = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      const i8 = new Int8Array(u8.buffer);
      const f = new Float32Array(i8.length);
      const s = e.scale;
      for (let i = 0; i < i8.length; i++) f[i] = i8[i] * s;
      out[name] = { data: f, shape: e.shape };
    }
    return out;
  }

  // ---- math helpers ----
  function mv(W, x, out, oLen, iLen, Woff) {
    for (let o = 0; o < oLen; o++) {
      let acc = 0;
      const base = (Woff || 0) + o * iLen;
      for (let i = 0; i < iLen; i++) acc += W[base + i] * x[i];
      out[o] = acc;
    }
  }
  function dot(a, b, n, ao, bo) {
    ao = ao || 0; bo = bo || 0;
    let s = 0;
    for (let i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
    return s;
  }
  function layerNorm(x, w, b, d) {
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[i];
    mean /= d;
    let varr = 0;
    for (let i = 0; i < d; i++) { const t = x[i] - mean; varr += t * t; }
    varr /= d;
    const inv = 1 / Math.sqrt(varr + LN_EPS);
    for (let i = 0; i < d; i++) x[i] = (x[i] - mean) * inv * w[i] + b[i];
  }
  function gelu(x, n) {
    const c = 0.7978845608028654; // sqrt(2/pi)
    for (let i = 0; i < n; i++) {
      const v = x[i];
      const t = c * (v + 0.044715 * v * v * v);
      x[i] = 0.5 * v * (1 + Math.tanh(t));
    }
  }
  function softmax(a, n, ao) {
    ao = ao || 0;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (a[ao + i] > mx) mx = a[ao + i];
    let sum = 0;
    for (let i = 0; i < n; i++) { a[ao + i] = Math.exp(a[ao + i] - mx); sum += a[ao + i]; }
    for (let i = 0; i < n; i++) a[ao + i] /= sum;
  }

  // ---- scratch buffers (module-level, reused every token) ----
  const _h = new Float32Array(D);
  const _qkv = new Float32Array(3 * D);
  const _att = new Float32Array(D);
  const _proj = new Float32Array(D);
  const _f1 = new Float32Array(FF);
  const _f2 = new Float32Array(D);
  const _x = new Float32Array(D);

  // ---- KV cache ----
  function makeCache() {
    const c = { k: [], v: [], scores: new Float32Array(MAXSEQ) };
    for (let l = 0; l < L; l++) { c.k.push([]); c.v.push([]); }
    return c;
  }

  // ---- forward pass for ONE token at position pos; returns logits [V] ----
  function forwardToken(tokId, pos, W, cache) {
    const tokEmb = W["tok_emb.weight"].data;
    const posEmb = W["pos_emb.weight"].data;
    for (let i = 0; i < D; i++) _x[i] = tokEmb[tokId * D + i] + posEmb[pos * D + i];

    for (let l = 0; l < L; l++) {
      const p = "blocks." + l + ".";
      // --- causal self-attention ---
      for (let i = 0; i < D; i++) _h[i] = _x[i];
      layerNorm(_h, W[p + "ln1.weight"].data, W[p + "ln1.bias"].data, D);
      mv(W[p + "attn.qkv.weight"].data, _h, _qkv, 3 * D, D, 0);
      const kL = cache.k[l], vL = cache.v[l];
      kL.push(_qkv.subarray(D, 2 * D).slice());
      vL.push(_qkv.subarray(2 * D, 3 * D).slice());
      const T = kL.length;
      for (let h = 0; h < H; h++) {
        const qoff = h * HD;
        const sc = cache.scores;
        for (let t = 0; t < T; t++) sc[t] = dot(_qkv, kL[t], HD, qoff, qoff) * HEAD_SCALE;
        softmax(sc, T, 0);
        const ooff = h * HD;
        for (let d = 0; d < HD; d++) _att[ooff + d] = 0;
        for (let t = 0; t < T; t++) {
          const w = sc[t];
          const v = vL[t];
          for (let d = 0; d < HD; d++) _att[ooff + d] += w * v[qoff + d];
        }
      }
      mv(W[p + "attn.proj.weight"].data, _att, _proj, D, D, 0);
      for (let i = 0; i < D; i++) _x[i] += _proj[i];
      // --- gelu feed-forward ---
      for (let i = 0; i < D; i++) _h[i] = _x[i];
      layerNorm(_h, W[p + "ln2.weight"].data, W[p + "ln2.bias"].data, D);
      mv(W[p + "ff.fc1.weight"].data, _h, _f1, FF, D, 0);
      gelu(_f1, FF);
      mv(W[p + "ff.fc2.weight"].data, _f1, _f2, D, FF, 0);
      for (let i = 0; i < D; i++) _x[i] += _f2[i];
    }
    layerNorm(_x, W["ln_f.weight"].data, W["ln_f.bias"].data, D);
    // weight-tied head: logits[j] = x · tokEmb[j]
    const logits = new Float32Array(V);
    for (let j = 0; j < V; j++) logits[j] = dot(_x, tokEmb, D, 0, j * D);
    return logits;
  }

  // ---- temperature + top-k sampling ----
  function sample(logits, temp, topK) {
    for (let i = 0; i < V; i++) logits[i] /= temp;
    const k = Math.min(topK, V);
    const idx = Array.from({ length: V }, (_, i) => i);
    idx.sort((a, b) => logits[b] - logits[a]);
    const top = idx.slice(0, k);
    const mx = logits[top[0]];
    let sum = 0;
    const p = new Float32Array(k);
    for (let i = 0; i < k; i++) { p[i] = Math.exp(logits[top[i]] - mx); sum += p[i]; }
    let r = Math.random() * sum;
    for (let i = 0; i < k; i++) { r -= p[i]; if (r <= 0) return top[i]; }
    return top[k - 1];
  }

  // ---- tokenizer ----
  function makeVocab(chars) {
    const stoi = {};
    for (let i = 0; i < chars.length; i++) stoi[chars[i]] = i;
    return { stoi, itos: chars.slice() };
  }
  function encode(text, vocab) {
    const ids = [];
    for (const ch of text) if (vocab.stoi[ch] !== undefined) ids.push(vocab.stoi[ch]);
    return ids;
  }

  /* =====================================================================
     Generation engine — a persistent session with KV cache. Used by BOTH
     the browser UI and the Node tests, so there is exactly one copy of the
     sliding-window logic. step(n) generates n tokens and returns them.
     ===================================================================== */
  function createSession(W, vocab) {
    let history = [];
    let cache = makeCache();
    let pos = 0;
    let lastLogits = null;

    function feed(t) {
      if (pos >= MAXSEQ) {
        // slide window: keep last half, rebuild cache
        cache = makeCache();
        pos = 0;
        const keep = Math.floor(MAXSEQ / 2);
        const recent = history.slice(history.length - keep);
        for (const t2 of recent) {
          lastLogits = forwardToken(t2, pos, W, cache);
          pos++;
        }
      }
      lastLogits = forwardToken(t, pos, W, cache);
      history.push(t);
      pos++;
    }

    return {
      prime(text) {
        history = encode(text, vocab);
        if (history.length === 0) history = [0];
        if (history.length > MAXSEQ) history = history.slice(history.length - MAXSEQ);
        cache = makeCache();
        pos = 0;
        lastLogits = null;
        // prompt is capped at MAXSEQ so no slide can trigger here
        for (const t of history) {
          lastLogits = forwardToken(t, pos, W, cache);
          pos++;
        }
      },
      step(nTokens, temp, topK) {
        const out = [];
        for (let n = 0; n < nTokens; n++) {
          const nextId = sample(lastLogits, temp, topK);
          out.push(nextId);
          feed(nextId);
        }
        return out;
      },
      decode(ids) { return ids.map(i => vocab.itos[i]).join(""); },
      get history() { return history; },
    };
  }

  // ---- module exports (Node) / window attach (browser) ----
  const api = {
    decodeWeights, forwardToken, sample, makeVocab, encode,
    createSession, makeCache,
    consts: { D, L, H, HD, FF, V, MAXSEQ },
  };
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.TinyMind = api;
  }

  /* =====================================================================
     Browser UI wiring
     ===================================================================== */
  if (typeof document !== "undefined") {
    let MODEL = null;
    let BUSY = false;
    let STOP = false;

    async function loadModel() {
      const st = document.getElementById("loadstatus");
      try {
        st.textContent = "loading model…";
        let raw = null;
        // 1) try a single bundled file (Netlify / local drag-drop deploy)
        try {
          const r = await fetch("model_weights.json");
          if (r.ok) raw = await r.json();
        } catch (e) {}
        // 2) else fetch chunked parts (GitHub Pages mirror, where the 2.4MB
        //    model is split into ~12 small files served same-origin)
        if (!raw) {
          st.textContent = "fetching model parts…";
          const r0 = await fetch("model_part_00.json");
          if (!r0.ok) throw new Error("no model_weights.json and no model_part_00.json");
          const p0 = await r0.json();
          const total = p0.total;
          raw = {
            config: p0.config, vocab: p0.vocab, n_params: p0.n_params,
            best_val_loss: p0.best_val_loss, quantization: p0.quantization,
            weights: Object.assign({}, p0.weights),
          };
          for (let i = 1; i < total; i++) {
            const r = await fetch("model_part_" + String(i).padStart(2, "0") + ".json");
            if (!r.ok) throw new Error("missing part " + i);
            const p = await r.json();
            Object.assign(raw.weights, p.weights);
            st.textContent = "fetching model parts… " + (i + 1) + "/" + total;
          }
        }
        st.textContent = "dequantizing int8 → float32…";
        await new Promise(r => setTimeout(r, 10));
        const W = decodeWeights(raw.weights);
        const vocab = makeVocab(raw.vocab.chars);
        MODEL = { W, vocab, valLoss: raw.best_val_loss };
        if (raw.best_val_loss) document.getElementById("vl").textContent = Number(raw.best_val_loss).toFixed(3);
        st.classList.remove("pulse");
        st.style.color = "var(--accent)";
        st.textContent = "✓ model ready — all " + raw.n_params.toLocaleString() + " params live in your browser";
        document.getElementById("go").disabled = false;
      } catch (e) {
        if (String(e.message).indexOf("no model_weights.json") !== -1) {
          // this mirror doesn't bundle the 2.3MB trained weights — point the
          // visitor to the one-drag publish instead of showing a raw error
          st.style.color = "var(--warm)";
          st.textContent = "";
          const info = document.createElement("span");
          info.innerHTML = "ℹ This live mirror doesn't bundle the 2.3MB trained weights. " +
            "To run the model, take <b>tinymind-gpt-site.zip</b> and drag it onto " +
            "<a href=\"https://app.netlify.com/drop\" target=\"_blank\" rel=\"noopener\">app.netlify.com/drop</a> — " +
            "you get a fully working URL in seconds.";
          st.appendChild(info);
        } else {
          st.style.color = "#f08080";
          st.textContent = "✗ failed to load: " + e.message;
        }
      }
    }

    // progressive generation driver (yields to the UI thread so the
    // typewriter effect stays smooth and Stop stays responsive)
    async function generateUI() {
      if (BUSY) { STOP = true; return; }
      if (!MODEL) return;
      const prompt = document.getElementById("prompt").value;
      const temp = parseFloat(document.getElementById("temp").value);
      const topK = parseInt(document.getElementById("topk").value);
      const maxNew = parseInt(document.getElementById("len").value);
      const out = document.getElementById("out");
      const btn = document.getElementById("go");
      const { W, vocab } = MODEL;

      BUSY = true; STOP = false;
      btn.textContent = "■ Stop";
      btn.classList.add("stop");
      out.innerHTML = "";
      const promptSpan = document.createElement("span");
      promptSpan.className = "prompt-part";
      promptSpan.textContent = prompt;
      const genSpan = document.createElement("span");
      const cursor = document.createElement("span");
      cursor.className = "cursor";
      cursor.innerHTML = "&nbsp;";
      out.append(promptSpan, genSpan, cursor);

      const session = createSession(W, vocab);
      session.prime(prompt);
      const t0 = performance.now();
      let count = 0, text = "";

      // generate in small chunks, yielding to the UI between chunks so the
      // typewriter animation stays smooth and the Stop button stays live
      while (count < maxNew && !STOP) {
        const chunk = Math.min(4, maxNew - count);
        const ids = session.step(chunk, temp, topK);
        text += session.decode(ids);
        genSpan.textContent = text;
        count += chunk;
        await new Promise(r => setTimeout(r, 0));
      }

      const dt = (performance.now() - t0) / 1000;
      if (count > 0) document.getElementById("spd").textContent = (count / dt).toFixed(0) + " tok/s";
      cursor.remove();
      BUSY = false;
      btn.textContent = "⚡ Generate";
      btn.classList.remove("stop");
    }

    document.getElementById("go").disabled = true;
    document.getElementById("go").addEventListener("click", generateUI);
    loadModel();
  }
})(typeof window !== "undefined" ? window : globalThis);
