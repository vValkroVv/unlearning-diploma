# Counterfactual generation API notes

The original operator notes for API-assisted counterfactual generation are kept
under `docs/operator_logs/openai_generation_counterfactuals_original.md`.

For public reproduction, configure a local or remote OpenAI-compatible endpoint
through environment variables instead of hardcoding credentials or machine
paths:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_API_KEY=EMPTY
export VLLM_MODEL=/path/to/generator/model
```

Then run the DUET or RWKU preparation scripts documented in
`docs/diploma_repro.md`. Do not commit API keys, private model names, generated
sidecars, or full counterfactual artifacts to Git.
