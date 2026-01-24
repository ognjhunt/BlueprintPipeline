#!/usr/bin/env python3
"""
Startup validation utilities for pipeline jobs.

Centralized credential and environment validation to fail fast
before jobs start executing.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

from tools.geniesim_adapter.config import (
    DEFAULT_GENIESIM_PORT,
    get_geniesim_host,
    get_geniesim_port,
)
from tools.config.production_mode import resolve_pipeline_environment, resolve_production_mode


class ValidationError(Exception):
    """Raised when startup validation fails."""
    pass


def validate_genie_sim_credentials(required: bool = False) -> Dict[str, any]:
    """
    Validate Genie Sim local framework configuration.

    Genie Sim runs locally via gRPC, no API credentials required.

    Args:
        required: Ignored (no credentials needed for local operation)

    Returns:
        Dict with validation results
    """
    warnings = []

    # Check for gRPC host/port configuration
    grpc_host = get_geniesim_host()
    grpc_port = get_geniesim_port()
    geniesim_root = os.getenv("GENIESIM_ROOT", "/opt/geniesim")
    environment = resolve_pipeline_environment()

    if grpc_host != "localhost":
        warnings.append(f"GENIESIM_HOST set to {grpc_host} (typical: localhost)")
    if (
        grpc_port == DEFAULT_GENIESIM_PORT
        and environment not in {"development", "dev", "local"}
    ):
        warnings.append(
            "GENIESIM_PORT is using the adapter default while running outside local/dev; "
            "set GENIESIM_PORT to the gRPC port exposed by your deployment."
        )

    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "grpc_host": grpc_host,
        "grpc_port": grpc_port,
        "geniesim_root": geniesim_root,
    }


def validate_gcs_credentials() -> Dict[str, any]:
    """
    Validate GCS credentials and bucket access.

    Verify GCS credentials work by testing bucket access.

    Returns:
        Dict with validation results
    """
    bucket = os.getenv("BUCKET", "")
    errors = []
    warnings = []

    if not bucket:
        warnings.append("BUCKET environment variable not set")

    # Check for GCP credentials
    gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    service_account = os.getenv("GCP_SERVICE_ACCOUNT")

    if not gcp_creds and not service_account:
        warnings.append("No GCP credentials found (GOOGLE_APPLICATION_CREDENTIALS or GCP_SERVICE_ACCOUNT)")

    # Try to test bucket access
    bucket_accessible = False
    try:
        from google.cloud import storage
        client = storage.Client()
        if bucket:
            bucket_obj = client.bucket(bucket)
            bucket_accessible = bucket_obj.exists()
    except ImportError:
        warnings.append("google-cloud-storage not installed - cannot verify bucket access")
    except Exception as e:
        warnings.append(f"Cannot access GCS bucket '{bucket}': {e}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "bucket_accessible": bucket_accessible,
    }


def validate_gemini_credentials(required: bool = False) -> Dict[str, any]:
    """
    Validate Gemini API credentials.

    Verify GEMINI_API_KEY is set if LLM features enabled.

    Args:
        required: If True, raise ValidationError if credentials missing

    Returns:
        Dict with validation results

    Raises:
        ValidationError: If required=True and validation fails
    """
    api_key = os.getenv("GEMINI_API_KEY")
    errors = []
    warnings = []

    if not api_key:
        if required:
            errors.append("GEMINI_API_KEY not set but LLM features enabled")
        else:
            warnings.append("GEMINI_API_KEY not set - LLM features will be disabled")
    elif len(api_key) < 10:
        errors.append("GEMINI_API_KEY appears invalid (too short)")

    if required and errors:
        error_msg = "\n".join([
            "Gemini credential validation failed:",
            *[f"  - {e}" for e in errors],
        ])
        raise ValidationError(error_msg)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "api_key_set": bool(api_key),
    }


def validate_firebase_credentials(required: bool = False) -> Dict[str, Any]:
    """
    Validate Firebase Storage credentials.

    Ensure required environment variables are present and credentials are usable.

    Args:
        required: If True, treat missing Firebase config as errors

    Returns:
        Dict with validation results
    """
    if (os.getenv("FIREBASE_UPLOAD_MODE", "") or "").strip().lower() == "local":
        local_dir = os.getenv("FIREBASE_UPLOAD_LOCAL_DIR")
        return {
            "valid": True,
            "errors": [],
            "warnings": [
                "FIREBASE_UPLOAD_MODE=local enabled; skipping Firebase credential validation."
            ],
            "local_upload_dir": local_dir,
        }

    bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
    emulator_host = os.getenv("FIREBASE_STORAGE_EMULATOR_HOST")
    service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")

    errors: List[str] = []
    warnings: List[str] = []

    if not bucket:
        message = "FIREBASE_STORAGE_BUCKET not set"
        if required:
            errors.append(message)
        else:
            warnings.append(message)

    service_account_required = not emulator_host
    has_service_account = bool(service_account_json or service_account_path)

    if service_account_required and not has_service_account:
        message = (
            "Firebase service account credentials not set "
            "(FIREBASE_SERVICE_ACCOUNT_JSON or FIREBASE_SERVICE_ACCOUNT_PATH)"
        )
        if required:
            errors.append(message)
        else:
            warnings.append(message)

    if service_account_json:
        try:
            json.loads(service_account_json)
        except json.JSONDecodeError as exc:
            errors.append(f"FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON: {exc}")

    if service_account_path:
        path = Path(service_account_path)
        if not path.exists():
            errors.append(f"FIREBASE_SERVICE_ACCOUNT_PATH does not exist: {path}")
        elif not path.is_file():
            errors.append(f"FIREBASE_SERVICE_ACCOUNT_PATH is not a file: {path}")

    should_init = bool(bucket) and (emulator_host or has_service_account)
    if should_init and not errors:
        try:
            from tools.firebase_upload.uploader import init_firebase

            init_firebase()
        except Exception as exc:
            errors.append(f"Firebase initialization failed: {exc}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "bucket_set": bool(bucket),
        "emulator_host_set": bool(emulator_host),
        "service_account_set": has_service_account,
    }


def validate_all_credentials(
    require_geniesim: bool = False,
    require_gemini: bool = False,
    validate_gcs: bool = True,
    validate_firebase: bool = False,
    require_firebase: bool = False,
) -> Dict[str, any]:
    """
    Validate all credentials at job startup.

    Comprehensive credential validation before job execution.

    Args:
        require_geniesim: Raise error if Genie Sim credentials missing
        require_gemini: Raise error if Gemini credentials missing
        validate_gcs: Check GCS bucket access
        validate_firebase: Include Firebase validation in the report
        require_firebase: Treat Firebase credentials as required when validating

    Returns:
        Dict with all validation results

    Raises:
        ValidationError: If any required credentials are invalid
    """
    results = {
        "geniesim": validate_genie_sim_credentials(required=require_geniesim),
        "gemini": validate_gemini_credentials(required=require_gemini),
    }

    if validate_gcs:
        results["gcs"] = validate_gcs_credentials()

    if validate_firebase:
        results["firebase"] = validate_firebase_credentials(required=require_firebase)

    # Collect all errors
    all_errors = []
    all_warnings = []
    for service, result in results.items():
        all_errors.extend([f"[{service}] {e}" for e in result.get("errors", [])])
        all_warnings.extend([f"[{service}] {w}" for w in result.get("warnings", [])])

    return {
        "valid": len(all_errors) == 0,
        "errors": all_errors,
        "warnings": all_warnings,
        "results": results,
    }


def print_validation_report(validation_result: Dict[str, any], job_name: str = "JOB") -> None:
    """
    Print a validation report.

    Args:
        validation_result: Result from validate_all_credentials()
        job_name: Name of the job for logging
    """
    print(f"\n[{job_name}] " + "=" * 60)
    print(f"[{job_name}] CREDENTIAL VALIDATION")
    print(f"[{job_name}] " + "=" * 60)

    if validation_result["valid"]:
        print(f"[{job_name}] ✅ All required credentials validated")
    else:
        print(f"[{job_name}] ❌ Credential validation FAILED")

    if validation_result["errors"]:
        print(f"[{job_name}] ")
        print(f"[{job_name}] ERRORS:")
        for error in validation_result["errors"]:
            print(f"[{job_name}]   - {error}")

    if validation_result["warnings"]:
        print(f"[{job_name}] ")
        print(f"[{job_name}] WARNINGS:")
        for warning in validation_result["warnings"]:
            print(f"[{job_name}]   - {warning}")

    print(f"[{job_name}] " + "=" * 60 + "\n")


def validate_and_fail_fast(
    job_name: str,
    require_geniesim: bool = False,
    require_gemini: bool = False,
    validate_gcs: bool = True,
    validate_firebase: bool = False,
    require_firebase: bool = False,
) -> None:
    """
    Validate credentials and exit if validation fails.

    Fail fast helper for job entry points.

    Args:
        job_name: Name of the job for logging
        require_geniesim: Raise error if Genie Sim credentials missing
        require_gemini: Raise error if Gemini credentials missing
        validate_gcs: Check GCS bucket access
        validate_firebase: Include Firebase validation in the report
        require_firebase: Treat Firebase credentials as required when validating

    Raises:
        SystemExit: If validation fails
    """
    skip_flag = os.getenv("SKIP_STARTUP_VALIDATION", "").strip().lower()
    if skip_flag in {"1", "true", "yes", "y"}:
        # Prevent skipping validation in production mode
        if resolve_production_mode():
            raise ValidationError(
                "Cannot skip startup validation in production mode. "
                "Set PIPELINE_ENV to 'development' or remove SKIP_STARTUP_VALIDATION."
            )
        print(f"[{job_name}] ⚠️ Skipping startup validation (SKIP_STARTUP_VALIDATION={skip_flag})")
        return

    try:
        result = validate_all_credentials(
            require_geniesim=require_geniesim,
            require_gemini=require_gemini,
            validate_gcs=validate_gcs,
            validate_firebase=validate_firebase,
            require_firebase=require_firebase,
        )

        print_validation_report(result, job_name)

        if not result["valid"]:
            print(f"[{job_name}] ❌ Exiting due to credential validation failures")
            sys.exit(1)

    except ValidationError as e:
        print(f"\n[{job_name}] ❌ VALIDATION ERROR:\n{e}\n")
        sys.exit(1)
