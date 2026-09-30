# Local testing

Experiment with Forecats providers outside Home Assistant.

```bash
cd local_testing
cp .env.example .env   # fill in keys / URLs / providers
uv run test.py
```

Environment highlights:

| Variable | Purpose |
| --- | --- |
| `ACTIVITY_PROVIDER` | `gemini`, `openai`, `invokeai`, or `unsloth` |
| `IMAGE_PROVIDER` | `gemini`, `invokeai`, or `unsloth` |
| `OUTPUT_DIR` | Directory for `forecats_original.png` / `forecats_optimized.png` |
| `GEMINI_API_KEY` | Required when either provider is Gemini |
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `*_MODEL` / `*_TIMEOUT` | OpenAI-compatible activity settings |
| `INVOKEAI_BASE_URL` / `*_MODEL` / `*_TIMEOUT` | InvokeAI settings (CPU image timeout default 3600s) |
| `UNSLOTH_BASE_URL` / `UNSLOTH_API_KEY` / `*_MODEL` / `*_TIMEOUT` | Unsloth Studio settings |

Requires [uv](https://docs.astral.sh/uv/).
