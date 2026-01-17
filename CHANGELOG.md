# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stream manifest parsing for episode downloads to handle large manifests efficiently.
- LRU metadata cache for the Vertex AI vector store to improve repeated lookups.
- Shared environment flag parser helper for consistent CLI flag handling.

### Changed
- Make quality gate approval waits cancelable to improve workflow control.
- Stream SQLite vector store queries to reduce memory pressure.
- Batch pgvector upserts to improve write throughput.

### Documentation
- Expanded environment variable documentation for pipeline configuration.
