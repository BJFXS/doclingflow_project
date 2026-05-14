# Docker Workflow

This repository is expected to run in Docker.

## Main Conversion Run

Use:

```bash
./run_with_docker.sh
```

This script:

- builds the image
- removes the previous run container
- starts a fresh conversion container
- mounts inputs and outputs

## Full Docker Test Run

Use:

```bash
./run_tests_with_docker.sh
```

This script:

- creates a fresh Docker test container
- runs the CLI smoke test
- runs the full `unittest` suite

## Representative Regression Run

Use:

```bash
./run_representative_with_docker.sh
```

This script uses a smaller mixed-format sample set for quicker regression checks.

## Heavy PDF Validation Run

Use:

```bash
./run_heavy_pdf_with_docker.sh
```

This script isolates:

- image-heavy PDFs
- scan-heavy PDFs
- long PDFs
- two-column PDFs

Use it when validating the expensive PDF routes separately from the faster mixed-format regression pass.
