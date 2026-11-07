# K3 Abliterator

A desktop app (PyQt6, Ubuntu) that finds the refusals a Kimi K3 model still makes and removes
them by **directional ablation** ("abliteration"). You give it prompts the model refuses and
prompts it answers. It works out the direction in the model's hidden state that separates the
two, and removes it while the model runs. If you're happy with the result, it can write the
change into a copy of the weights.

Kimi K3 has 2.8 trillion parameters (1.5 TB on disk), so it runs through
[colibri](https://github.com/JustVugg/colibri), which streams experts from disk. Tools that load
the whole model into memory (transformers, TransformerLens, Heretic, FailSpy's abliterator) can't
run it on a desktop. This app drives colibri instead.

## How it works

- **Engine patch.** `src/k3_abliterator/engine/` holds a small patch to colibri's Kimi K3
  engine (`k3_dirabl.h` and `kimi_k3.patch`). It is off unless `K3_ABL_CTL` names a control file,
  which the engine re-reads at the start of every request. The app builds its own copy of the
  engine into `~/.local/share/k3-abliterator/engine/` and runs it through `coli serve` with
  `COLI_ENGINE`. Your `~/colibri/c/kimi_k3` is never rebuilt or changed.
- **Capture.** For each prompt the engine dumps the hidden state that every layer reads, at the
  last prompt token. The direction at each layer is the difference between the mean of the
  refused prompts and the mean of the normal ones. Its score is how well it separates prompts it
  wasn't computed from.
- **Live switch.** The engine removes the chosen directions from the embedding and from every
  attention and MLP output in the chosen layer range. K3 replaces the residual stream with a
  mix of block sums (AttnRes), but each of those is a sum of the same outputs, so the direction
  is gone from everything the layers read. Captures are taken with the ablation on, so running
  the capture again finds the direction that is left over.
- **Bake.** Removing a direction from those outputs is the same as projecting it out of the
  matrices that produce them: `embed_tokens`, `o_proj`, the routed experts' up-projection,
  `shared_experts.down_proj` and the first layer's `down_proj`. All of them are BF16, about
  32 GB for the whole model; the MXFP4 experts aren't touched. The baked folder links every
  original file and adds one shard with the changed tensors.

## Using it

1. **Start engine.** It loads the model through colibri, which takes a few minutes. Only one
   Kimi K3 server fits in memory, so the app offers to stop another one if it finds one running.
   The engine runs as a user service (`k3-abliterator-engine`) with no memory cap: colibri sizes
   its own memory from what the machine has free, as when you run `coli` yourself.
2. **Prompts.** Add 24 or more prompts the model still refuses, in different wordings. The
   normal set comes filled in.
3. **Directions → Capture prompts and compute.** At colibri's measured ~2 s per prompt token,
   expect an hour or two for 60 prompts. Captures are kept, so changing a few prompts only
   reads those again. Pick a layer (the best middle-of-stack one is preselected), then
   **Add this direction**.
4. **Test.** Ask the same prompt with the ablation off and on, or run the refusal check over
   every refused prompt. Generation runs at several seconds a token.
5. Repeat step 3 to find what's left, adjust the strength or the layer range, then **Bake**.

The baked model runs like any other: `cd ~/colibri && ./c/coli web --model <baked folder>`.

## Install

```bash
scripts/install.sh
```

This needs [uv](https://docs.astral.sh/uv/), colibri in `~/colibri` (or `COLIBRI_DIR`), built
once with `make kimi_k3`, and a Kimi K3 checkpoint. The app checks this repo's GitHub releases
for updates. To run from a checkout instead: `PYTHONPATH=src python3 -m k3_abliterator`.

## Problem reports

**Help → Report a problem** sends, along with your note: the app's log, the engine's own status
lines from its log (lines like `[K3] ...` and `[ABL] ...`, never request text), the app version,
the OS version, the number of directions and the strength and layer settings. Uncaught errors
are sent the same way. Prompts, answers and captures are never sent. Reports go to the
developer's own server.

## Tests

```bash
python3 -m pytest tests/                                          # maths, bake, control file, store
cc -O2 -o build/test_dirabl tests/test_dirabl.c -lm && build/test_dirabl build/t   # engine header
QT_QPA_PLATFORM=offscreen python3 tests/e2e_gui.py build/e2e      # whole GUI flow against a fake engine
```

## License

MIT. `kimi_k3.patch` modifies colibri's `kimi_k3.c` and is under colibri's Apache-2.0 license.
