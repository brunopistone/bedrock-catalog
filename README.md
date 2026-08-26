# bedrock-catalog

A CLI for exploring the catalog of models available in **Amazon Bedrock**.

## Which API it calls

Model discovery lives on the Amazon Bedrock **control plane** — the
[`bedrock`](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock.html) service in
boto3 (`boto3.client('bedrock')`):

- `list-models` → [`ListFoundationModels`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListFoundationModels.html)
- `get-model` → [`GetFoundationModel`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_GetFoundationModel.html)
- both also → [`ListInferenceProfiles`](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_ListInferenceProfiles.html),
  for the IDs that actually invoke `INFERENCE_PROFILE` models (see
  [below](#base-model-ids-vs-inference-profile-ids))

## Quick Start

```bash
# Install
uv tool install .

# Authenticate with any AWS credentials (no API key needed)
export AWS_PROFILE=my-profile
export AWS_REGION=us-east-1

# List the models in the region
bedrock-catalog list-models
```

Example output:

```
Region: us-east-1

Available Models (122):
------------------------------------------------------------
  ID: openai.gpt-oss-120b-1:0
      Name: gpt-oss-120b
      Provider: OpenAI
      Modalities: TEXT -> TEXT
      Streaming: yes
      Inference: ON_DEMAND
      Customization: none
      Status: ACTIVE

  ID: anthropic.claude-haiku-4-5-20251001-v1:0
      Name: Claude Haiku 4.5
      Provider: Anthropic
      Modalities: TEXT, IMAGE -> TEXT
      Streaming: yes
      Inference: INFERENCE_PROFILE
      Invoke as (required): global.anthropic.claude-haiku-4-5-20251001-v1:0, us.anthropic.claude-haiku-4-5-20251001-v1:0
      Customization: none
      Status: ACTIVE
```

## Configuration

There is no API key — authentication is standard AWS SigV4. Credentials resolve in this order:

1. `--profile <name>` — use that named AWS profile
2. Default AWS credential chain (`AWS_PROFILE`, SSO, env vars, instance profiles, ...)

The region comes from `--region`, `AWS_REGION` / `AWS_DEFAULT_REGION`, or the profile's configured
region. If none of those yield a region, the CLI fails with a clear message rather than guessing.

### Using a `.env` file

```bash
cp .env.example .env
# Edit .env with your values
```

### Required IAM permissions

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:ListFoundationModels",
    "bedrock:GetFoundationModel",
    "bedrock:ListInferenceProfiles"
  ],
  "Resource": "*"
}
```

Without `bedrock:ListInferenceProfiles` the CLI still lists models — it prints a warning and omits
the invocable IDs.

## Commands

### List models

```bash
# All models in the region
bedrock-catalog list-models

# Filter server-side (all four filters are ListFoundationModels parameters)
bedrock-catalog list-models --provider Anthropic
bedrock-catalog list-models --output-modality IMAGE
bedrock-catalog list-models --inference-type ON_DEMAND
bedrock-catalog list-models --inference-type INFERENCE_PROFILE
bedrock-catalog list-models --customization-type FINE_TUNING

# Another region / profile, or JSON output
bedrock-catalog list-models --region eu-west-1 --profile my-profile
bedrock-catalog list-models --json
```

### Get one model

```bash
bedrock-catalog get-model openai.gpt-oss-120b-1:0
bedrock-catalog get-model anthropic.claude-sonnet-4-20250514-v1:0 --json
```

Control plane IDs are **versioned** (`openai.gpt-oss-120b-1:0`), unlike the short IDs used by
`bedrock-mantle` (`openai.gpt-oss-120b`).

Pass the **base** model ID here — inference profile IDs (`us.*`, `global.*`) are not valid input for
`GetFoundationModel`; they are reported in the output as the IDs to invoke with.

## Base model IDs vs inference profile IDs

`ListFoundationModels` reports the **base** model ID. When a model supports `INFERENCE_PROFILE` and
**not** `ON_DEMAND`, that base ID **cannot be invoked** — `Converse` / `InvokeModel` require a
cross-region inference profile ID instead:

|                         |                                                   |
| ----------------------- | ------------------------------------------------- |
| Base ID (not invocable) | `anthropic.claude-haiku-4-5-20251001-v1:0`        |
| Invoke as               | `us.anthropic.claude-haiku-4-5-20251001-v1:0`     |
| Invoke as               | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |

The prefix is the routing scope — `us` / `eu` / `apac` route within that geography, `global` routes
worldwide — and a model may offer several. These IDs come from `ListInferenceProfiles`, and appear as
`Invoke as` in the output:

- **`Invoke as (required)`** — the model has no on-demand option, so you _must_ use one of these IDs.
- **`Invoke as`** — the model also supports `ON_DEMAND`, so the profile is an alternative (it adds
  cross-region capacity).

In `us-east-1` at the time of writing, 47 of 121 models support `INFERENCE_PROFILE` and 41 of those
have no on-demand fallback, so the base ID alone would fail for roughly a third of the catalog. Those
counts drift as models ship — recount them with:

```bash
bedrock-catalog list-models --inference-type INFERENCE_PROFILE --json | jq length
```

With `--json`, each model carries an added `inferenceProfileIds` array (empty when no profile exists):

```bash
bedrock-catalog list-models --inference-type INFERENCE_PROFILE --json
```

## Provisioned-throughput variants

Some IDs carry a context-window suffix — `amazon.nova-pro-v1:0:24k`,
`amazon.nova-lite-v1:0:300k`, `amazon.nova-2-lite-v1:0:256k`. These report
`Inference: PROVISIONED` with neither `ON_DEMAND` nor `INFERENCE_PROFILE`, and they are **not
separately invocable**: they exist to be purchased as Provisioned Throughput at a fixed context
size. For on-demand inference use the unsuffixed base ID (`amazon.nova-pro-v1:0`) or its inference
profile.

18 of the 121 models in `us-east-1` are in this category, which is why the catalog looks larger than
the set you can actually call.

### Discovery info

```bash
bedrock-catalog info
```

## What the API does and does not report

Reported per model: `modelId`, `modelArn`, `modelName`, `providerName`, input/output modalities,
`responseStreamingSupported`, `inferenceTypesSupported`, `customizationsSupported`, and
`modelLifecycle` (`ACTIVE` or `LEGACY`, plus launch date). The invocable `us.*` / `global.*` IDs are
**not** part of this response — the CLI adds them from `ListInferenceProfiles`.

**Not reported** — by either this API or the `bedrock-mantle` `/v1/models` endpoint:

- context window size
- maximum output tokens
- pricing

Those are published in the [model documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
only. To count tokens for a specific request, use the `bedrock-runtime` `CountTokens` API.

Nor is per-model **API and feature support** reported, which varies by model generation and can only
be established by calling:

|                            |                                                                                                                                                                                                                                                               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Converse structured output | `outputConfig.textFormat` (JSON schema) is accepted by Claude 4.5/4.6 but **rejected** by the Claude `*-5` generation, which returns `ValidationException: output_config.format: Extra inputs are not permitted`. Fall back to requesting JSON in the prompt. |
| Reasoning parameters       | Mutually exclusive per family and rejected across families. Claude 4.6+ takes `thinking={"type":"adaptive"}` + `output_config.effort`; Claude 4.5 and older take `thinking={"type":"enabled","budget_tokens":N}`; other families take `reasoning_effort`.     |
| `temperature`              | Deprecated on newer models — Claude `opus-5` and `opus-4-8` return `400 \`temperature\` is deprecated for this model`.                                                                                                                                        |
| Mantle API route           | On the `bedrock-mantle` endpoint each model is served on exactly one of `/v1/chat/completions`, `/openai/v1/responses` or `/anthropic/v1/messages`. See `bedrock-mantle list-models`.                                                                         |

`ListFoundationModels` also returns deprecated IDs kept for backwards compatibility — check the
`Status` field and prefer `ACTIVE` models.

## Shell Completion

```bash
bedrock-catalog completion bash --install   # or: zsh, fish
```

Then open a new shell. To enable it only for the current shell:

```bash
eval "$(bedrock-catalog completion bash)"
```

## Development

```bash
# Install dependencies
uv sync

# Run only the unit tests (no credentials needed)
uv run pytest tests/ -v -m unit

# Run everything, including tests that call the real API
uv run pytest tests/ -v

# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Run all pre-commit checks
uv run pre-commit run --all-files
```

## License

MIT
