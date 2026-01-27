# Checksums HMAC key rotation

## Purpose

`CHECKSUMS_HMAC_KEY` and `CHECKSUMS_HMAC_KEY_ID` protect the integrity of bundle
artifacts by signing `checksums.json` with an HMAC-SHA256 signature. The key
itself is stored in Secret Manager and injected at runtime. The optional key ID
labels which key signed a bundle so auditors can correlate signatures during
rotation.

## Signing and verification flow

1. **Bundle creation (Genie Sim import)**
   - `checksums.json` is generated with SHA-256 hashes for bundle files.
   - If `CHECKSUMS_HMAC_KEY` is set, the importer signs the checksums payload and
     adds a `signature` stanza. If `CHECKSUMS_HMAC_KEY_ID` is set, it is included
     in the signature payload as `key_id`.
2. **Import verification**
   - The importer validates `checksums.json` against the bundle contents and
     fails the import if any checksum mismatches are detected.
3. **Delivery integrity audit**
   - The delivery audit job verifies the `checksums.json` signature. In
     production mode, a missing signature is treated as a failure.

## Rotation steps

1. **Generate a new HMAC key**
   - Use a cryptographically secure random generator (e.g., 256-bit key).
   - Record the new key ID you plan to use (date-based or Secret Manager
     version).
2. **Update Secret Manager**
   - Add the new key as a new Secret Manager version.
   - Update configuration so workloads can reference the new version.
3. **Roll out the new key**
   - Update runtime configuration to set `CHECKSUMS_HMAC_KEY` and
     `CHECKSUMS_HMAC_KEY_ID` to the new values.
   - Deploy the Genie Sim import job so new bundles are signed with the new key.
4. **Dual-key window (optional)**
   - If downstream auditors or consumers need to validate older bundles, keep
     the previous key available in Secret Manager for the duration of the
     validation window.
   - If you need dual-key verification, update the verification job to try the
     old key first and then the new key (or vice versa).
5. **Verify rollout**
   - Run a new import and confirm `checksums.json` includes the new `key_id`.
   - Run the delivery integrity audit and confirm the signature passes.
6. **Retire the old key**
   - Remove the old key from Secret Manager once all consumers have moved.

## Validation and error signatures

### Import checksum validation

The import job validates `checksums.json` content and will fail imports when
checksums are missing or mismatched. Typical error signatures in logs:

- `Checksum verification failed.`
- `checksums.json missing 'files' mapping for verification`
- `checksums.json signature mismatch` (if signature verification is enabled in
  a downstream audit)

### Signature validation

The delivery integrity audit verifies the signature. Typical error signatures:

- `checksums.json signature is required in production`
- `checksums.json signature is missing`
- `CHECKSUMS_HMAC_KEY is required to verify signature`
- `checksums.json signature value is missing`
- `Unsupported signature algorithm: <value>`
- `checksums.json signature mismatch`

When any of these appear, confirm the correct Secret Manager version is mapped
into `CHECKSUMS_HMAC_KEY`, the expected `CHECKSUMS_HMAC_KEY_ID` is set, and the
bundle’s `checksums.json` was generated after the rotation cutover.
