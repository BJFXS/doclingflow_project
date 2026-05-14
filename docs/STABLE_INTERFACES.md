# Stable Interfaces

This file defines the interfaces that should be treated as stable for normal use.

## Stable User Entry Points

- `./run_with_docker.sh`
- `./run_tests_with_docker.sh`
- `./run_representative_with_docker.sh`
- `./run_heavy_pdf_with_docker.sh`
- `./run_install_check_with_docker.sh`
- `./run_build_check_with_docker.sh`
- `./run_twine_check_with_docker.sh`
- `./run_dist_install_check_with_docker.sh`

## Stable CLI Entry Points

- `doclingflow --help`
- `doclingflow convert`
- `doclingflow batch`
- `doclingflow inspect`
- `doclingflow doctor`
- `doclingflow version`

## Stable Python Entry Points

- `doclingflow.convert_file`
- `doclingflow.convert_batch`
- `doclingflow.inspect_file`

## Stable Package Namespaces

These package paths should be preferred over direct root-level imports:

- `doclingflow.adapters`
- `doclingflow.analyzers`
- `doclingflow.benchmarks`
- `doclingflow.pipeline`
- `doclingflow.processors`
- `doclingflow.utils`
- `doclingflow.document_types`

## Compatibility Layer

Root-level modules still exist and are still used internally.

They are currently treated as compatibility and implementation modules, not as the preferred long-term user-facing import path. New external integrations should prefer `doclingflow.*` imports.
