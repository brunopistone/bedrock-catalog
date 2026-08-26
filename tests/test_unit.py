"""
Unit tests for the bedrock-catalog CLI.

These tests require no credentials and make no API calls - boto3 is stubbed out
so the region resolution, filter mapping, output rendering and error mapping can
be checked in isolation.

Run with: uv run pytest tests/ -v -m unit
"""

import json

import click
import pytest

from main import (
    COMPLETE_VAR,
    PROG_NAME,
    call_api,
    create_client,
    fetch_inference_profiles,
    generate_completion,
    modality_summary,
    resolve_region,
)

SAMPLE_MODEL = {
    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-oss-120b-1:0",
    "modelId": "openai.gpt-oss-120b-1:0",
    "modelName": "gpt-oss-120b",
    "providerName": "OpenAI",
    "inputModalities": ["TEXT"],
    "outputModalities": ["TEXT"],
    "responseStreamingSupported": True,
    "customizationsSupported": [],
    "inferenceTypesSupported": ["ON_DEMAND"],
    "modelLifecycle": {"status": "ACTIVE"},
}

SECOND_MODEL = {
    "modelId": "amazon.titan-image-generator-v1",
    "modelName": "Titan Image Generator",
    "providerName": "Amazon",
    "inputModalities": ["TEXT", "IMAGE"],
    "outputModalities": ["IMAGE"],
    "responseStreamingSupported": False,
    "customizationsSupported": ["FINE_TUNING"],
    "inferenceTypesSupported": ["ON_DEMAND", "PROVISIONED"],
    "modelLifecycle": {"status": "LEGACY"},
}

# Invocable only through an inference profile - the base ID below will not work.
PROFILE_MODEL = {
    "modelId": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "modelName": "Claude Haiku 4.5",
    "providerName": "Anthropic",
    "inputModalities": ["TEXT", "IMAGE"],
    "outputModalities": ["TEXT"],
    "responseStreamingSupported": True,
    "customizationsSupported": [],
    "inferenceTypesSupported": ["INFERENCE_PROFILE"],
    "modelLifecycle": {"status": "ACTIVE"},
}

# Has a profile, but on-demand works too - the profile is an alternative.
DUAL_MODEL = {
    "modelId": "amazon.nova-pro-v1:0",
    "modelName": "Nova Pro",
    "providerName": "Amazon",
    "inputModalities": ["TEXT"],
    "outputModalities": ["TEXT"],
    "responseStreamingSupported": True,
    "customizationsSupported": [],
    "inferenceTypesSupported": ["ON_DEMAND", "INFERENCE_PROFILE"],
    "modelLifecycle": {"status": "ACTIVE"},
}

HAIKU_ID = PROFILE_MODEL["modelId"]

# Mirrors the real shape: global.* profiles carry a region-less ARN alongside
# the regional one, and each model may map to several profiles.
PROFILE_SUMMARIES = [
    {
        "inferenceProfileId": f"us.{HAIKU_ID}",
        "type": "SYSTEM_DEFINED",
        "models": [
            {"modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{HAIKU_ID}"},
            {"modelArn": f"arn:aws:bedrock:us-west-2::foundation-model/{HAIKU_ID}"},
        ],
    },
    {
        "inferenceProfileId": f"global.{HAIKU_ID}",
        "type": "SYSTEM_DEFINED",
        "models": [
            {"modelArn": f"arn:aws:bedrock:::foundation-model/{HAIKU_ID}"},
            {"modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{HAIKU_ID}"},
        ],
    },
    {
        "inferenceProfileId": "us.amazon.nova-pro-v1:0",
        "type": "SYSTEM_DEFINED",
        "models": [
            {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"},
        ],
    },
]


class FakeSession:
    """Minimal stand-in for a boto3 Session."""

    def __init__(self, region_name=None, profile_name=None):
        self.region_name = region_name
        self.profile_name = profile_name


class FakeMeta:
    def __init__(self, region_name):
        self.region_name = region_name


class FakePaginator:
    """Replays canned pages, recording the paginate() parameters."""

    def __init__(self, client, pages):
        self._client = client
        self._pages = pages

    def paginate(self, **params):
        self._client.calls["list_inference_profiles"] = params
        for page in self._pages:
            yield {"inferenceProfileSummaries": page}


class FakeClient:
    """Records the calls made against it and replays canned responses."""

    def __init__(
        self,
        region_name="us-east-1",
        models=None,
        details=None,
        profile_pages=None,
        profiles_error=None,
    ):
        self.meta = FakeMeta(region_name)
        self.calls = {}
        self._models = (
            models
            if models is not None
            else [SAMPLE_MODEL, SECOND_MODEL, PROFILE_MODEL, DUAL_MODEL]
        )
        self._details = details if details is not None else SAMPLE_MODEL
        # Split across pages so the pagination loop is exercised.
        self._profile_pages = (
            profile_pages
            if profile_pages is not None
            else [PROFILE_SUMMARIES[:2], PROFILE_SUMMARIES[2:]]
        )
        self._profiles_error = profiles_error

    def list_foundation_models(self, **params):
        self.calls["list_foundation_models"] = params
        return {"modelSummaries": self._models}

    def get_foundation_model(self, **params):
        self.calls["get_foundation_model"] = params
        return {"modelDetails": self._details}

    def get_paginator(self, operation_name):
        self.calls["get_paginator"] = operation_name
        if self._profiles_error is not None:
            raise self._profiles_error
        return FakePaginator(self, self._profile_pages)


@pytest.fixture
def patch_client(monkeypatch):
    """Return a helper installing a given FakeClient behind create_client()."""

    def _patch(client):
        captured = {}

        def _create_client(profile=None, region=None):
            captured["profile"] = profile
            captured["region"] = region
            return client

        monkeypatch.setattr("main.create_client", _create_client)
        client.captured = captured
        return client

    return _patch


@pytest.fixture
def fake_client(patch_client):
    """Patch create_client() so commands run against a FakeClient."""
    return patch_client(FakeClient())


# =============================================================================
# Region Resolution
# =============================================================================


@pytest.mark.unit
@pytest.mark.usefixtures("clean_env")
class TestResolveRegion:
    """Tests for resolve_region()."""

    def test_explicit_region_wins(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")

        assert resolve_region(FakeSession(region_name="ap-south-1"), "us-east-2") == "us-east-2"

    def test_env_region_used(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")

        assert resolve_region(FakeSession(region_name="ap-south-1"), None) == "eu-west-1"

    def test_default_region_env_used(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")

        assert resolve_region(FakeSession(), None) == "eu-central-1"

    def test_falls_back_to_session_region(self):
        assert resolve_region(FakeSession(region_name="ap-south-1"), None) == "ap-south-1"

    def test_missing_region_raises(self):
        with pytest.raises(click.ClickException, match="Could not determine an AWS region"):
            resolve_region(FakeSession(), None)


@pytest.mark.unit
@pytest.mark.usefixtures("clean_env")
class TestCreateClient:
    """Tests for create_client()."""

    def test_builds_bedrock_client_for_profile_and_region(self, monkeypatch):
        captured = {}

        class Session(FakeSession):
            def client(self, service_name, region_name=None):
                captured["service_name"] = service_name
                captured["region_name"] = region_name
                return FakeClient(region_name=region_name)

        def _build_aws_session(profile):
            captured["profile"] = profile
            return Session(profile_name=profile)

        monkeypatch.setattr("main.build_session", _build_aws_session)

        create_client(profile="my-profile", region="eu-west-2")

        assert captured["profile"] == "my-profile"
        assert captured["service_name"] == "bedrock"
        assert captured["region_name"] == "eu-west-2"


# =============================================================================
# Error Mapping
# =============================================================================


@pytest.mark.unit
class TestCallApi:
    """Tests for call_api() turning AWS errors into ClickExceptions."""

    def test_passes_through_response(self):
        assert call_api(lambda **kw: {"ok": kw}, byProvider="Amazon") == {
            "ok": {"byProvider": "Amazon"}
        }

    def test_client_error_becomes_click_exception(self):
        from botocore.exceptions import ClientError

        def _raise():
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
                "ListFoundationModels",
            )

        with pytest.raises(click.ClickException, match="AccessDeniedException: not authorized"):
            call_api(_raise)

    def test_credential_error_becomes_click_exception(self):
        from botocore.exceptions import NoCredentialsError

        def _raise():
            raise NoCredentialsError()

        with pytest.raises(click.ClickException, match="AWS error"):
            call_api(_raise)


# =============================================================================
# Rendering Helpers
# =============================================================================


@pytest.mark.unit
class TestModalitySummary:
    """Tests for modality_summary()."""

    def test_single_modalities(self):
        assert modality_summary(SAMPLE_MODEL) == "TEXT -> TEXT"

    def test_multiple_input_modalities(self):
        assert modality_summary(SECOND_MODEL) == "TEXT, IMAGE -> IMAGE"

    def test_missing_modalities(self):
        assert modality_summary({}) == "- -> -"


# =============================================================================
# Inference Profiles
# =============================================================================


@pytest.mark.unit
class TestFetchInferenceProfiles:
    """Tests for fetch_inference_profiles() mapping base model IDs to profiles."""

    def test_maps_base_model_id_to_profiles(self):
        profiles = fetch_inference_profiles(FakeClient())

        assert profiles[HAIKU_ID] == [f"global.{HAIKU_ID}", f"us.{HAIKU_ID}"]

    def test_requests_only_system_defined_profiles(self):
        client = FakeClient()

        fetch_inference_profiles(client)

        assert client.calls["get_paginator"] == "list_inference_profiles"
        assert client.calls["list_inference_profiles"] == {"typeEquals": "SYSTEM_DEFINED"}

    def test_matches_region_less_and_regional_arns(self):
        """global.* profiles list both ARN shapes for the same model."""
        client = FakeClient(profile_pages=[[PROFILE_SUMMARIES[1]]])

        assert fetch_inference_profiles(client) == {HAIKU_ID: [f"global.{HAIKU_ID}"]}

    def test_ignores_models_without_profiles(self):
        assert "openai.gpt-oss-120b-1:0" not in fetch_inference_profiles(FakeClient())

    def test_access_denied_degrades_gracefully(self, capsys):
        from botocore.exceptions import ClientError

        error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
            "ListInferenceProfiles",
        )

        assert fetch_inference_profiles(FakeClient(profiles_error=error)) == {}
        assert "could not list inference profiles" in capsys.readouterr().err


@pytest.mark.unit
class TestInvokeAsRendering:
    """Tests for how profile IDs surface in the model output."""

    def test_profile_only_model_marks_ids_required(self, run_cli, fake_client):
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert f"Invoke as (required): global.{HAIKU_ID}, us.{HAIKU_ID}" in result.output

    def test_on_demand_model_lists_profile_as_alternative(self, run_cli, fake_client):
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert "Invoke as: us.amazon.nova-pro-v1:0" in result.output

    def test_model_without_profile_has_no_invoke_line(self, run_cli, patch_client):
        patch_client(FakeClient(models=[SAMPLE_MODEL]))

        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert "Invoke as" not in result.output

    def test_get_model_shows_required_profile_ids(self, run_cli, patch_client):
        patch_client(FakeClient(details=PROFILE_MODEL))

        result = run_cli(["get-model", HAIKU_ID])

        assert result.exit_code == 0
        assert f"Invoke as (required): global.{HAIKU_ID}, us.{HAIKU_ID}" in result.output

    def test_get_model_json_includes_profile_ids(self, run_cli, patch_client):
        patch_client(FakeClient(details=PROFILE_MODEL))

        result = run_cli(["get-model", HAIKU_ID, "--json"])

        assert result.exit_code == 0
        assert json.loads(result.output)["inferenceProfileIds"] == [
            f"global.{HAIKU_ID}",
            f"us.{HAIKU_ID}",
        ]


# =============================================================================
# list-models Command
# =============================================================================


@pytest.mark.unit
class TestListModelsCommand:
    """Tests for the 'list-models' command."""

    def test_lists_models(self, run_cli, fake_client):
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert "Region: us-east-1" in result.output
        assert "Available Models (4):" in result.output
        assert "ID: openai.gpt-oss-120b-1:0" in result.output
        assert "Provider: OpenAI" in result.output
        assert "Modalities: TEXT -> TEXT" in result.output
        assert "Streaming: yes" in result.output
        assert "Customization: none" in result.output
        assert "Status: ACTIVE" in result.output

    def test_renders_second_model_fields(self, run_cli, fake_client):
        result = run_cli(["list-models"])

        assert "Streaming: no" in result.output
        assert "Inference: ON_DEMAND, PROVISIONED" in result.output
        assert "Customization: FINE_TUNING" in result.output
        assert "Status: LEGACY" in result.output

    def test_no_filters_sends_no_params(self, run_cli, fake_client):
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert fake_client.calls["list_foundation_models"] == {}
        assert "Filters:" not in result.output

    def test_filters_map_to_api_params(self, run_cli, fake_client):
        result = run_cli(
            [
                "list-models",
                "--provider",
                "Anthropic",
                "--output-modality",
                "IMAGE",
                "--inference-type",
                "ON_DEMAND",
                "--customization-type",
                "FINE_TUNING",
            ]
        )

        assert result.exit_code == 0
        assert fake_client.calls["list_foundation_models"] == {
            "byProvider": "Anthropic",
            "byOutputModality": "IMAGE",
            "byInferenceType": "ON_DEMAND",
            "byCustomizationType": "FINE_TUNING",
        }
        assert "Filters: byCustomizationType=FINE_TUNING" in result.output

    def test_profile_and_region_forwarded(self, run_cli, fake_client):
        result = run_cli(["list-models", "--profile", "my-profile", "--region", "eu-west-1"])

        assert result.exit_code == 0
        assert fake_client.captured == {"profile": "my-profile", "region": "eu-west-1"}

    def test_json_output(self, run_cli, fake_client):
        result = run_cli(["list-models", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert [m["modelId"] for m in payload] == [
            "openai.gpt-oss-120b-1:0",
            "amazon.titan-image-generator-v1",
            HAIKU_ID,
            "amazon.nova-pro-v1:0",
        ]

    def test_json_output_includes_profile_ids(self, run_cli, fake_client):
        result = run_cli(["list-models", "--json"])

        assert result.exit_code == 0
        profiles = {m["modelId"]: m["inferenceProfileIds"] for m in json.loads(result.output)}
        assert profiles[HAIKU_ID] == [f"global.{HAIKU_ID}", f"us.{HAIKU_ID}"]
        assert profiles["amazon.nova-pro-v1:0"] == ["us.amazon.nova-pro-v1:0"]
        # No profile exists for this one, but the key is always present.
        assert profiles["openai.gpt-oss-120b-1:0"] == []

    def test_accepts_inference_profile_filter(self, run_cli, fake_client):
        result = run_cli(["list-models", "--inference-type", "INFERENCE_PROFILE"])

        assert result.exit_code == 0
        assert fake_client.calls["list_foundation_models"] == {
            "byInferenceType": "INFERENCE_PROFILE"
        }

    def test_rejects_invalid_modality(self, run_cli, fake_client):
        result = run_cli(["list-models", "--output-modality", "VIDEO"])

        assert result.exit_code != 0


# =============================================================================
# get-model Command
# =============================================================================


@pytest.mark.unit
class TestGetModelCommand:
    """Tests for the 'get-model' command."""

    def test_shows_model_details(self, run_cli, fake_client):
        result = run_cli(["get-model", "openai.gpt-oss-120b-1:0"])

        assert result.exit_code == 0
        assert fake_client.calls["get_foundation_model"] == {
            "modelIdentifier": "openai.gpt-oss-120b-1:0"
        }
        assert "ID: openai.gpt-oss-120b-1:0" in result.output
        assert "ARN: arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-oss-120b-1:0" in (
            result.output
        )

    def test_json_output(self, run_cli, fake_client):
        result = run_cli(["get-model", "openai.gpt-oss-120b-1:0", "--json"])

        assert result.exit_code == 0
        assert json.loads(result.output)["modelName"] == "gpt-oss-120b"

    def test_requires_model_id(self, run_cli):
        result = run_cli(["get-model"])

        assert result.exit_code != 0


# =============================================================================
# info / help / completion
# =============================================================================


@pytest.mark.unit
class TestInfoCommand:
    """Tests for the 'info' command."""

    def test_explains_endpoint_split(self, run_cli):
        result = run_cli(["info"])

        assert result.exit_code == 0
        assert "boto3.client('bedrock')" in result.output
        assert "bedrock-mantle" in result.output
        assert "bedrock-runtime" in result.output
        assert "ListFoundationModels" in result.output
        assert "GetFoundationModel" in result.output

    def test_documents_missing_characteristics(self, run_cli):
        result = run_cli(["info"])

        assert result.exit_code == 0
        assert "WHAT IT DOES NOT REPORT" in result.output
        assert "context window size" in result.output
        assert "maximum output tokens" in result.output

    def test_documents_authentication(self, run_cli):
        result = run_cli(["info"])

        assert result.exit_code == 0
        assert "AUTHENTICATION" in result.output
        assert "--profile" in result.output


@pytest.mark.unit
class TestHelpCommands:
    """Tests for help output."""

    def test_main_help(self, run_cli):
        result = run_cli(["--help"])

        assert result.exit_code == 0
        assert "list-models" in result.output
        assert "get-model" in result.output
        assert "info" in result.output
        assert "chat" not in result.output

    def test_list_models_help(self, run_cli):
        result = run_cli(["list-models", "--help"])

        assert result.exit_code == 0
        for option in ("--provider", "--output-modality", "--inference-type", "--json"):
            assert option in result.output


@pytest.mark.unit
class TestCompletion:
    """Tests for completion script generation."""

    @pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
    def test_generate_completion(self, shell):
        script = generate_completion(shell)

        assert COMPLETE_VAR in script
        assert PROG_NAME in script

    def test_complete_var_matches_click_derivation(self):
        """COMPLETE_VAR must be exactly what Click derives from PROG_NAME.

        Click looks up `_{PROG_NAME}_COMPLETE` - dashes and dots to underscores,
        uppercased - when the shell asks for completions. If our constant drifts from
        that, the installed script exports one variable while Click reads another, and
        completion stops working with no error on either side. Renaming the CLI is the
        way that drift happens.
        """
        derived = f"_{PROG_NAME.replace('-', '_').replace('.', '_')}_COMPLETE".upper()

        assert derived == COMPLETE_VAR

    def test_generate_completion_unsupported_shell(self):
        with pytest.raises(click.ClickException, match="Unsupported shell"):
            generate_completion("powershell")

    def test_completion_command_prints_script(self, run_cli):
        result = run_cli(["completion", "bash"])

        assert result.exit_code == 0
        assert COMPLETE_VAR in result.output
