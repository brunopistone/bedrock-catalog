#!/usr/bin/env python3
"""
CLI for exploring the models available in Amazon Bedrock.

Model discovery lives on the Amazon Bedrock control plane - the `bedrock`
service in boto3 - which is what this CLI calls:

- ListFoundationModels: every foundation model offered in the region
- GetFoundationModel: details for a single model
- ListInferenceProfiles: the us.*/global.* IDs needed to invoke models that
  only support INFERENCE_PROFILE, since the base model ID will not work

Neither inference endpoint offers this: `bedrock-runtime` only exposes inference
operations (Converse, InvokeModel, CountTokens, ...), while `bedrock-mantle` has
its own OpenAI-compatible GET /v1/models.

Authentication is standard AWS SigV4 - there is no API key. Credentials come
from --profile or the default credential chain.

See: https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock.html
"""

import json
import os
from typing import Any

import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The boto3 service that owns model discovery (the Bedrock control plane).
SERVICE_NAME = "bedrock"

# Filter values accepted by the ListFoundationModels API.
OUTPUT_MODALITIES = ["TEXT", "IMAGE", "EMBEDDING"]
INFERENCE_TYPES = ["ON_DEMAND", "PROVISIONED", "INFERENCE_PROFILE"]
CUSTOMIZATION_TYPES = ["FINE_TUNING", "CONTINUED_PRE_TRAINING", "DISTILLATION"]

# Maps a CLI option name to the ListFoundationModels request parameter.
LIST_FILTERS = {
    "provider": "byProvider",
    "output_modality": "byOutputModality",
    "inference_type": "byInferenceType",
    "customization_type": "byCustomizationType",
}


def build_session(profile: str | None):
    """Build a boto3 session for the given profile (or the default chain).

    Imports boto3 lazily to keep CLI startup fast for --help and info.
    """
    try:
        import boto3
    except ImportError as e:
        raise click.ClickException(
            f"Could not import '{e.name}'. Reinstall the CLI to restore AWS dependencies:\n"
            "  uv tool install . --force   (or: pip install -e .)"
        ) from None

    return boto3.Session(profile_name=profile)


def resolve_region(session, region: str | None) -> str:
    """Determine which region to query, preferring an explicit --region.

    Args:
        session: A boto3 Session whose configured region is the fallback.
        region: Region passed on the command line, if any.

    Returns:
        The region name to use.
    """
    resolved = (
        region
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or session.region_name
    )

    if not resolved:
        raise click.ClickException(
            "Could not determine an AWS region.\n"
            "Pass --region, set AWS_REGION/AWS_DEFAULT_REGION, or configure a\n"
            "region for the profile in ~/.aws/config."
        )

    return resolved


def create_client(profile: str | None = None, region: str | None = None):
    """Create the Amazon Bedrock control-plane client used for model discovery.

    Args:
        profile: AWS named profile to authenticate with, or None for the
            default credential chain.
        region: Region to query, or None to resolve it from the environment
            or the profile.

    Returns:
        A boto3 'bedrock' client.
    """
    session = build_session(profile)
    resolved_region = resolve_region(session, region)

    try:
        return session.client(SERVICE_NAME, region_name=resolved_region)
    except Exception as e:
        raise click.ClickException(f"Could not create a Bedrock client: {e}") from None


def call_api(api_call, **params) -> dict[str, Any]:
    """Invoke a Bedrock API call, turning AWS errors into clean CLI errors.

    Args:
        api_call: The bound boto3 client method to call.
        **params: Request parameters for the call.

    Returns:
        The parsed API response.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        return api_call(**params)
    except ClientError as e:
        error = e.response.get("Error", {})
        raise click.ClickException(
            f"{error.get('Code', 'AWSError')}: {error.get('Message', str(e))}"
        ) from None
    except BotoCoreError as e:
        # Covers missing/expired credentials, endpoint resolution failures, etc.
        raise click.ClickException(f"AWS error: {e}") from None


def fetch_inference_profiles(client) -> dict[str, list[str]]:
    """Map each base model ID to the inference profile IDs that can invoke it.

    Models whose inferenceTypesSupported is INFERENCE_PROFILE cannot be called
    with the base model ID returned by ListFoundationModels - they need a
    cross-region inference profile ID (us.*, eu.*, apac.*, global.*), which only
    ListInferenceProfiles reports.

    Only SYSTEM_DEFINED profiles are requested: application profiles are
    per-account copies and would not generalise as an "invoke as" hint.

    Args:
        client: The Bedrock control-plane client.

    Returns:
        Base model ID -> sorted profile IDs. Empty when the caller lacks
        bedrock:ListInferenceProfiles, so discovery still works without it.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    profiles: dict[str, list[str]] = {}

    try:
        pages = client.get_paginator("list_inference_profiles").paginate(
            typeEquals="SYSTEM_DEFINED"
        )
        summaries = [
            summary for page in pages for summary in page.get("inferenceProfileSummaries", [])
        ]
    except (ClientError, BotoCoreError) as e:
        click.echo(
            f"Warning: could not list inference profiles ({e}).\n"
            "         Invocable IDs for INFERENCE_PROFILE models are omitted.",
            err=True,
        )
        return profiles

    for summary in summaries:
        profile_id = summary.get("inferenceProfileId")
        if not profile_id:
            continue
        for model in summary.get("models") or []:
            # global.* profiles list a region-less ARN alongside the regional
            # one, so match on the model ID rather than the full ARN.
            arn = model.get("modelArn") or ""
            base_id = arn.rpartition("foundation-model/")[2]
            if base_id and profile_id not in profiles.setdefault(base_id, []):
                profiles[base_id].append(profile_id)

    # Alphabetical puts global.* before us.*, which is also the order to prefer.
    return {base_id: sorted(ids) for base_id, ids in profiles.items()}


def modality_summary(model: dict[str, Any]) -> str:
    """Render a model's input/output modalities as 'TEXT -> TEXT'."""
    inputs = ", ".join(model.get("inputModalities") or ["-"])
    outputs = ", ".join(model.get("outputModalities") or ["-"])
    return f"{inputs} -> {outputs}"


def echo_model(model: dict[str, Any], profile_ids: list[str] | None = None) -> None:
    """Print one model as an indented block.

    Args:
        model: A model summary or details payload.
        profile_ids: Inference profile IDs that can invoke this model, if any.
    """
    lifecycle = model.get("modelLifecycle") or {}
    customizations = model.get("customizationsSupported") or []
    inference_types = model.get("inferenceTypesSupported") or []

    click.echo(f"  ID: {model.get('modelId')}")
    click.echo(f"      Name: {model.get('modelName')}")
    click.echo(f"      Provider: {model.get('providerName')}")
    click.echo(f"      Modalities: {modality_summary(model)}")
    click.echo(f"      Streaming: {'yes' if model.get('responseStreamingSupported') else 'no'}")
    click.echo(f"      Inference: {', '.join(inference_types) or '-'}")

    if profile_ids:
        # Without ON_DEMAND the base model ID above cannot be invoked at all.
        required = "ON_DEMAND" not in inference_types
        label = "Invoke as (required)" if required else "Invoke as"
        click.echo(f"      {label}: {', '.join(profile_ids)}")

    click.echo(f"      Customization: {', '.join(customizations) or 'none'}")
    click.echo(f"      Status: {lifecycle.get('status', '-')}")
    click.echo()


@click.group()
def cli():
    """
    CLI for exploring the models available in Amazon Bedrock.

    Model discovery lives on the Amazon Bedrock control plane (the 'bedrock'
    service), which is what this CLI calls:

    - list-models: every foundation model in the region (ListFoundationModels)
    - get-model: details for one model (GetFoundationModel)

    Both also call ListInferenceProfiles, to report the inference profile IDs
    (us.*, global.*) required to invoke INFERENCE_PROFILE-only models.

    Authentication uses standard AWS credentials (SigV4) - no API key. Pass
    --profile to pick a named profile, or rely on the default credential chain
    (AWS_PROFILE, SSO, env vars, etc.).

    The region comes from --region, AWS_REGION/AWS_DEFAULT_REGION, or the
    profile's configured region.
    """
    pass


@cli.command("list-models")
@click.option(
    "--profile",
    default=None,
    help="AWS named profile to authenticate with.",
)
@click.option(
    "--region",
    default=None,
    help="Region to list models for (defaults to the environment or profile).",
)
@click.option(
    "--provider",
    default=None,
    help="Only show models from this provider (e.g. Anthropic, Meta).",
)
@click.option(
    "--output-modality",
    type=click.Choice(OUTPUT_MODALITIES),
    default=None,
    help="Only show models producing this output modality.",
)
@click.option(
    "--inference-type",
    type=click.Choice(INFERENCE_TYPES),
    default=None,
    help="Only show models supporting this inference type.",
)
@click.option(
    "--customization-type",
    type=click.Choice(CUSTOMIZATION_TYPES),
    default=None,
    help="Only show models supporting this customization type.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print the API response as JSON, plus an inferenceProfileIds key per model.",
)
def list_models(
    profile: str | None,
    region: str | None,
    provider: str | None,
    output_modality: str | None,
    inference_type: str | None,
    customization_type: str | None,
    as_json: bool,
):
    """
    List the foundation models available in a region.

    Calls the ListFoundationModels API. The response describes modalities,
    streaming and customization support - it does NOT include context window
    or token limits (see 'info').

    Models supporting INFERENCE_PROFILE also show the inference profile IDs
    (us.*, global.*, ...) that can invoke them, from ListInferenceProfiles.
    """
    client = create_client(profile, region)

    options = {
        "provider": provider,
        "output_modality": output_modality,
        "inference_type": inference_type,
        "customization_type": customization_type,
    }
    params = {LIST_FILTERS[name]: value for name, value in options.items() if value}

    response = call_api(client.list_foundation_models, **params)
    models = response.get("modelSummaries", [])
    profiles = fetch_inference_profiles(client)

    if as_json:
        enriched = [
            {**model, "inferenceProfileIds": profiles.get(model.get("modelId"), [])}
            for model in models
        ]
        click.echo(json.dumps(enriched, indent=2, default=str))
        return

    click.echo(f"Region: {client.meta.region_name}")
    if params:
        applied = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
        click.echo(f"Filters: {applied}")
    click.echo()

    click.echo(f"Available Models ({len(models)}):")
    click.echo("-" * 60)

    for model in models:
        echo_model(model, profiles.get(model.get("modelId")))


@cli.command("get-model")
@click.argument("model_id")
@click.option(
    "--profile",
    default=None,
    help="AWS named profile to authenticate with.",
)
@click.option(
    "--region",
    default=None,
    help="Region to query (defaults to the environment or profile).",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Print the API response as JSON, plus an inferenceProfileIds key.",
)
def get_model(model_id: str, profile: str | None, region: str | None, as_json: bool):
    """
    Show details for a single model.

    MODEL_ID is a Bedrock model identifier, for example
    anthropic.claude-sonnet-4-20250514-v1:0. Note that the control plane uses
    versioned IDs, which differ from the short IDs on the bedrock-mantle
    endpoint (openai.gpt-oss-120b-1:0 vs openai.gpt-oss-120b).

    Pass the base model ID: inference profile IDs (us.*, global.*) are not
    accepted here, they are reported as the IDs to invoke the model with.
    """
    client = create_client(profile, region)

    response = call_api(client.get_foundation_model, modelIdentifier=model_id)
    details = response.get("modelDetails", {})
    profile_ids = fetch_inference_profiles(client).get(details.get("modelId"), [])

    if as_json:
        click.echo(
            json.dumps({**details, "inferenceProfileIds": profile_ids}, indent=2, default=str)
        )
        return

    click.echo(f"Region: {client.meta.region_name}")
    click.echo()
    echo_model(details, profile_ids)
    click.echo(f"      ARN: {details.get('modelArn')}")

    lifecycle = details.get("modelLifecycle") or {}
    if lifecycle.get("startOfLifeTime"):
        click.echo(f"      Available since: {lifecycle['startOfLifeTime']}")


@cli.command("info")
def info():
    """
    Show how model discovery works in Amazon Bedrock.

    Explains which API this CLI calls, why, and which model characteristics
    the API does and does not report.
    """
    click.echo("""
Amazon Bedrock - model discovery
================================

WHICH API THIS CLI CALLS
------------------------
Model discovery lives on the Amazon Bedrock CONTROL PLANE - the 'bedrock'
service in boto3 (boto3.client('bedrock')):

  list-models   ->  ListFoundationModels + ListInferenceProfiles
  get-model     ->  GetFoundationModel  + ListInferenceProfiles

Neither inference endpoint provides this:

  bedrock-runtime   Inference only (Converse, InvokeModel, CountTokens, ...).
                    It has NO model listing API.
  bedrock-mantle    OpenAI-compatible. Lists its own models with
                    GET /v1/models, and that list differs from this one.

Reference:
  https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock.html

AUTHENTICATION
--------------
Standard AWS SigV4 - there is no API key. Credentials resolve in this order:

  1. --profile <name>  Use that named AWS profile
  2. Default AWS credential chain (AWS_PROFILE, SSO, env vars, ...)

The region comes from --region, AWS_REGION / AWS_DEFAULT_REGION, or the
profile's configured region.

WHAT THE API REPORTS
--------------------
For each model you get:

  modelId, modelArn, modelName, providerName
  inputModalities / outputModalities   (TEXT, IMAGE, EMBEDDING)
  responseStreamingSupported           (streaming inference)
  inferenceTypesSupported              (ON_DEMAND, PROVISIONED)
  customizationsSupported              (FINE_TUNING, ...)
  modelLifecycle                       (ACTIVE or LEGACY, launch date)

WHAT IT DOES NOT REPORT
-----------------------
Neither this API nor the bedrock-mantle /v1/models endpoint returns model
characteristics such as:

  - context window size
  - maximum output tokens
  - pricing

Those are published in the documentation only, per model:
  https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

To count tokens for a specific request against a model, use the
bedrock-runtime CountTokens API instead.

Nor does it report per-model API and FEATURE support, which varies by model
generation and can only be established by calling:

  Converse structured output    Passing outputConfig.textFormat (JSON schema)
                                is accepted by Claude 4.5/4.6 but rejected by
                                the Claude *-5 generation, which answers
                                ValidationException 'output_config.format:
                                Extra inputs are not permitted'. Fall back to
                                asking for JSON in the prompt.

  Reasoning parameters          Mutually exclusive per family, and rejected
                                across families. Claude 4.6+ takes
                                thinking={"type":"adaptive"} plus
                                output_config.effort; Claude 4.5 and older take
                                thinking={"type":"enabled","budget_tokens":N};
                                other families take reasoning_effort.

  temperature                   Deprecated on newer models - Claude opus-5 and
                                opus-4-8 answer HTTP 400 '`temperature` is
                                deprecated for this model'.

  Mantle API route              A model on the bedrock-mantle endpoint is served
                                on exactly one of /v1/chat/completions,
                                /openai/v1/responses or /anthropic/v1/messages.
                                See 'bedrock-mantle list-models'.

MODEL IDS
---------
Control plane IDs are versioned (openai.gpt-oss-120b-1:0), while the
bedrock-mantle endpoint uses short IDs (openai.gpt-oss-120b). The two lists
also differ in size - check the one matching the endpoint you will call.

BASE MODEL IDS vs INFERENCE PROFILE IDS
---------------------------------------
ListFoundationModels reports the BASE model ID. When a model lists
INFERENCE_PROFILE and not ON_DEMAND, that base ID CANNOT be invoked: Converse
and InvokeModel need a cross-region inference profile ID instead.

  base ID     anthropic.claude-haiku-4-5-20251001-v1:0     (not invocable)
  invoke as   us.anthropic.claude-haiku-4-5-20251001-v1:0
              global.anthropic.claude-haiku-4-5-20251001-v1:0

Those IDs come from a separate API, ListInferenceProfiles, which this CLI also
calls - they appear as 'Invoke as' in the output. The prefix selects the
routing scope (us, eu, apac = regional; global = worldwide), and a model may
offer several. Some ON_DEMAND models also have profiles, listed as an
alternative rather than a requirement.

LEGACY MODELS
-------------
ListFoundationModels also returns deprecated IDs kept for backwards
compatibility. Check modelLifecycle.status (the 'Status' field) and prefer
ACTIVE models.

PROVISIONED-THROUGHPUT VARIANTS
-------------------------------
Some IDs carry a context-window suffix and report Inference: PROVISIONED with
neither ON_DEMAND nor INFERENCE_PROFILE:

  amazon.nova-pro-v1:0:24k        amazon.nova-lite-v1:0:300k
  amazon.nova-2-lite-v1:0:256k    amazon.titan-embed-image-v1:0

These are NOT separately invocable - they exist to be purchased as Provisioned
Throughput at a fixed context size. For on-demand inference use the unsuffixed
base ID (amazon.nova-pro-v1:0) or its inference profile. 18 of the 121 models in
us-east-1 are in this category, which is why the catalog looks larger than the
set you can actually call.
""")


PROG_NAME = "bedrock-catalog"
# Click derives this from the invoked program name: uppercased, non-alphanumerics to
# underscores, with a leading one. It MUST track PROG_NAME - a mismatch breaks
# completion silently, with no error on either side.
COMPLETE_VAR = "_BEDROCK_CATALOG_COMPLETE"

# Per-shell config: the file to write the script to and the profile to source it from.
SHELL_CONFIG = {
    "bash": ("~/.config/bedrock-catalog/complete.bash", "~/.bashrc"),
    "zsh": ("~/.config/bedrock-catalog/complete.zsh", "~/.zshrc"),
    "fish": ("~/.config/fish/completions/bedrock-catalog.fish", None),
}


def generate_completion(shell: str) -> str:
    """Render the shell completion script for the given shell."""
    from click.shell_completion import get_completion_class

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    return comp_cls(cli, {}, PROG_NAME, COMPLETE_VAR).source()


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.option(
    "--install",
    is_flag=True,
    default=False,
    help="Write the completion script and wire it into your shell profile.",
)
def completion(shell: str, install: bool):
    """
    Generate (or install) shell completion for bedrock-catalog.

    \b
    Print the script to evaluate in the current shell:
      eval "$(bedrock-catalog completion bash)"

    \b
    Or install it permanently for new shells:
      bedrock-catalog completion bash --install
    """
    script = generate_completion(shell)

    if not install:
        click.echo(script)
        return

    script_path, profile = SHELL_CONFIG[shell]
    script_file = os.path.expanduser(script_path)
    os.makedirs(os.path.dirname(script_file), exist_ok=True)
    with open(script_file, "w") as f:
        f.write(script + "\n")
    click.echo(f"Wrote completion script to {script_path}")

    # fish auto-loads from its completions dir; nothing else to wire up.
    if profile is None:
        click.echo("Restart your shell (or open a new one) to enable completion.")
        return

    profile_file = os.path.expanduser(profile)
    source_line = f"[ -f {script_path} ] && source {script_path}"
    # Namespaced to PROG_NAME: a bare "# bedrock ..." marker also matches the line
    # a sibling bedrock CLI writes, so --install would skip wiring us up silently.
    marker = f"# {PROG_NAME} shell completion"

    existing = ""
    if os.path.exists(profile_file):
        with open(profile_file) as f:
            existing = f.read()

    if marker in existing:
        click.echo(f"{profile} already sources the completion script. Nothing to do.")
    else:
        with open(profile_file, "a") as f:
            f.write(f"\n{marker}\n{source_line}\n")
        click.echo(f"Added source line to {profile}")

    click.echo(f"Run 'source {profile}' or open a new shell to enable completion.")


if __name__ == "__main__":
    cli()
