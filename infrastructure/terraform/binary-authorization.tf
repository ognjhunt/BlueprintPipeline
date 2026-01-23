# =============================================================================
# Binary Authorization
# =============================================================================

locals {
  binary_authorization_attestors = length(var.binary_authorization_attestors) > 0 ? var.binary_authorization_attestors : (
    var.enable_binary_authorization && var.create_binary_authorization_attestor ? [
      google_binary_authorization_attestor.primary[0].name,
    ] : []
  )
}

resource "google_container_analysis_note" "binary_auth_attestor" {
  count   = var.enable_binary_authorization && var.create_binary_authorization_attestor ? 1 : 0
  project = var.project_id
  name    = "${var.binary_authorization_attestor_name}-note"

  attestation_authority {
    hint {
      human_readable_name = var.binary_authorization_attestor_description
    }
  }
}

resource "google_binary_authorization_attestor" "primary" {
  count   = var.enable_binary_authorization && var.create_binary_authorization_attestor ? 1 : 0
  name    = var.binary_authorization_attestor_name
  project = var.project_id

  attestation_authority_note {
    note_reference = google_container_analysis_note.binary_auth_attestor[0].name
  }

  public_keys {
    id                       = var.binary_authorization_attestor_public_key_id
    ascii_armored_pgp_public_key = var.binary_authorization_attestor_public_key
  }
}

resource "google_binary_authorization_policy" "default" {
  count   = var.enable_binary_authorization && var.create_binary_authorization_policy ? 1 : 0
  project = var.project_id

  admission_whitelist_patterns = var.binary_authorization_admission_whitelist

  default_admission_rule {
    evaluation_mode         = "REQUIRE_ATTESTATION"
    enforcement_mode        = "ENFORCED_BLOCK_AND_AUDIT_LOG"
    require_attestations_by = local.binary_authorization_attestors
  }
}
