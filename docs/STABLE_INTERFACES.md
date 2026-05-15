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

These package paths should be preferred over direct root-level imports.

Only the package-level exports are treated as stable. Deep submodule paths such as
`doclingflow.pipeline.task_executor` or `doclingflow.processors.image_handler`
are not part of the stable contract.

- `doclingflow.adapters`
- `doclingflow.analyzers`
- `doclingflow.benchmarks`
- `doclingflow.pipeline`
- `doclingflow.processors`
- `doclingflow.utils`
- `doclingflow.document_types`

## Compatibility Layer

Root-level modules still exist and are still used as the implementation source of truth.

The `doclingflow` package is the preferred user-facing tool surface, while the
root-level modules remain the active implementation modules during the staged
migration. New external integrations should prefer package-level `doclingflow.*`
imports rather than deep root-level imports.
