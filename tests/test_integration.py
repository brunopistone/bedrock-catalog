"""
Integration tests for the bedrock-catalog CLI.

These tests call the real Bedrock control plane and require AWS credentials
that allow bedrock:ListFoundationModels and bedrock:GetFoundationModel.
They are skipped automatically when no credentials resolve.

Run with: uv run pytest tests/ -v
Skip them: uv run pytest tests/ -v -m "not integration"
"""

import json

import pytest


@pytest.mark.integration
@pytest.mark.slow
class TestListModelsIntegration:
    """Tests for 'list-models' against the real API."""

    def test_lists_models(self, run_cli, aws_region):
        result = run_cli(["list-models", "--region", aws_region])

        assert result.exit_code == 0
        assert f"Region: {aws_region}" in result.output
        assert "Available Models (" in result.output
        assert "ID:" in result.output
        assert "Provider:" in result.output

    def test_json_output_is_parseable(self, run_cli, aws_region):
        result = run_cli(["list-models", "--region", aws_region, "--json"])

        assert result.exit_code == 0
        models = json.loads(result.output)
        assert models, "expected at least one model in the region"
        assert {"modelId", "modelArn", "providerName"} <= set(models[0])

    def test_provider_filter_is_applied(self, run_cli, aws_region):
        result = run_cli(["list-models", "--region", aws_region, "--provider", "Amazon", "--json"])

        assert result.exit_code == 0
        models = json.loads(result.output)
        assert {m["providerName"] for m in models} == {"Amazon"}

    def test_output_modality_filter_is_applied(self, run_cli, aws_region):
        result = run_cli(
            ["list-models", "--region", aws_region, "--output-modality", "EMBEDDING", "--json"]
        )

        assert result.exit_code == 0
        for model in json.loads(result.output):
            assert "EMBEDDING" in model["outputModalities"]


@pytest.mark.integration
@pytest.mark.slow
class TestInferenceProfileIntegration:
    """Tests that INFERENCE_PROFILE models report an invocable ID."""

    def test_profile_only_models_have_profile_ids(self, run_cli, aws_region):
        result = run_cli(["list-models", "--region", aws_region, "--json"])

        assert result.exit_code == 0
        profile_only = [
            m
            for m in json.loads(result.output)
            if "INFERENCE_PROFILE" in m["inferenceTypesSupported"]
            and "ON_DEMAND" not in m["inferenceTypesSupported"]
        ]
        assert profile_only, "expected models invocable only through a profile"

        for model in profile_only:
            ids = model["inferenceProfileIds"]
            assert ids, f"{model['modelId']} has no invocable inference profile ID"
            # The base ID is not usable, so every ID must carry a scope prefix.
            assert all(i.split(".", 1)[0] in {"us", "eu", "apac", "global"} for i in ids), ids

    def test_inference_profile_filter_is_accepted(self, run_cli, aws_region):
        result = run_cli(
            ["list-models", "--region", aws_region, "--inference-type", "INFERENCE_PROFILE"]
        )

        assert result.exit_code == 0
        assert "Invoke as (required):" in result.output


@pytest.mark.integration
@pytest.mark.slow
class TestGetModelIntegration:
    """Tests for 'get-model' against the real API."""

    def test_get_first_listed_model(self, run_cli, aws_region):
        listed = run_cli(["list-models", "--region", aws_region, "--json"])
        model_id = json.loads(listed.output)[0]["modelId"]

        result = run_cli(["get-model", model_id, "--region", aws_region, "--json"])

        assert result.exit_code == 0
        assert json.loads(result.output)["modelId"] == model_id

    def test_unknown_model_fails_cleanly(self, run_cli, aws_region):
        result = run_cli(["get-model", "does.not-exist-v9:0", "--region", aws_region])

        assert result.exit_code != 0
        # Surfaced as a CLI error, not a traceback
        assert "Error:" in result.output


@pytest.mark.integration
class TestRegionResolutionIntegration:
    """Tests for region handling against real credentials."""

    def test_invalid_region_fails_cleanly(self, run_cli, aws_region):
        result = run_cli(["list-models", "--region", "xx-nowhere-1"])

        assert result.exit_code != 0
        assert "Error:" in result.output
