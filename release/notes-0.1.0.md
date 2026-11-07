First release.

- Finds the refusal direction between prompts a Kimi K3 model still refuses and everyday prompts, at every layer, scored on held-out prompts.
- Removes it live in colibri through a small engine patch (built by the app; the stock engine isn't touched), with a strength slider and layer range. Capture again with it on to find what's left.
- Test tab: the same prompt with the ablation off and on, side by side, and a refusal check over every refused prompt.
- Bake: writes the ablation into a new model folder (one new shard of changed BF16 tensors, links to everything else).

Checked on the real model (tests/real_check.py): captures are written; with a direction switched on it is gone from every layer's input (largest share 1.0 to 8e-8); switching off gives the original back exactly; a layer range leaves earlier layers untouched. Not yet checked on the real model: removing an actual refusal, and the bake (checked against the engine's formula on a small synthetic model).
