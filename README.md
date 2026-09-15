# TinyMind GPT

A **real transformer language model — 1,810,944 parameters — trained entirely from scratch** in PyTorch, then exported to int8 and re-implemented in pure JavaScript so it runs 100% in your browser with no server and no API.

## What this is

- **Architecture:** decoder-only GPT (the same family as GPT-2/GPT-3) — token + position embeddings, 4× [causal self-attention (6 heads, d=192) + GELU feed-forward], weight-tied output head.
- **Size:** 1,810,944 parameters (≈1.81M, within the 1–10M target).
- **Tokenization:** character-level (vocab 70).
- **Training:** trained from scratch on ~1.1MB of Tiny Shakespeare + general-knowledge snippets, CPU-only (2 cores), ~2,000 AdamW steps with cosine LR schedule. Final validation loss ≈ 2.1 nats/char.
- **Quantization:** weights are symmetric int8-quantized per tensor → ~2.3MB total, dequantized to float32 in the browser at load time.
- **Inference:** a from-scratch JavaScript forward pass with **KV-caching** and a sliding context window, so generation streams token-by-token and can run indefinitely past the 128-char context.

## Files

| file | role |
|------|------|
| `index.html` | the UI: prompt, temperature / top-k / length controls, live generation |
| `model.js` | the pure-JS GPT engine (forward pass, sampling, KV cache) + UI wiring |
| `model_weights.json` | the trained weights (int8, base64) — **not in this repo**; bundled in the deploy zip |

## Run it

This is a static site — no build step.

1. Grab `model_weights.json` (from the deploy zip provided alongside this repo) and drop it next to `index.html` / `model.js`.
2. Open `index.html` in a browser (or host the folder anywhere static).

### Publish on Netlify (one step)

The deploy zip (`tinymind-gpt-site.zip`) contains all three files. Drag it onto
**https://app.netlify.com/drop** → you get a live `https://*.netlify.app` URL
in seconds. The model loads same-origin, so it works with zero configuration.

> Why a zip instead of a live link from me? This assistant's sandbox cannot reach
> Netlify (egress-blocked) and cannot stream a 2.3MB file through its own context
> into a host, so the model is delivered in the zip for a one-drag publish.

## Honest scope

At 1.81M parameters this is roughly **100,000× smaller** than frontier models
like GPT-4 or Claude, and it trained for minutes, not months. It speaks
recognisable, Shakespeare-flavoured English — it will *not* answer questions like
a chatbot. It exists to show, end-to-end, that a real GPT can be built, trained,
and run in the browser from scratch.

## Reproduce training

`train_model.py` trains the model and exports `model_weights.json` (int8).
The JS engine in `model.js` was verified against the PyTorch model's logits
(argmax match, top-5 overlap 5/5, max |Δ| ≈ 0.03 from int8 quantization).
