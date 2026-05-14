# doclingflow_project

`doclingflow_project` is a Docker-first document-to-Markdown tool built on top of [Docling](https://github.com/docling-project/docling).

Its goal is not just to convert files quickly. The project is optimized for a stricter target:

- convert common document formats into Markdown for LLM workflows
- preserve as much source information as possible
- apply strategy-aware handling for difficult PDFs instead of treating all files the same

## What It Is

This project is a high-fidelity conversion layer on top of Docling.

It currently supports common source types such as:

- `pdf`
- `docx`
- `xlsx`
- `pptx`
- `html`
- `htm`
- `md`
- `txt`
- `jpg`
- `jpeg`
- `png`
- `tiff`
- `bmp`

The project adds:

- file-type and PDF layout analysis
- strategy-aware routing
- long PDF chunking
- scan-aware OCR handling
- Markdown structure repair
- image reference normalization
- CSV / JSON batch reports

## Tool Entry Points

The project now exposes a package-style CLI:

```bash
doclingflow --help
doclingflow convert input.pdf -o output.md
doclingflow batch /data/input -o /data/output
doclingflow inspect input.pdf
doclingflow doctor
```

The same functionality is also exposed through the Python package entrypoint:

```bash
python -m doclingflow --help
```

## Recommended Runtime

This repository is intended to run in Docker.

The normal end-to-end path is:

1. `run_with_docker.sh`
2. `Dockerfile`
3. `python -m doclingflow batch /data/input -o /data/output`

## Standard Docker Run

Run the bundled input set:

```bash
./run_with_docker.sh
```

Run with custom input and output directories:

```bash
./run_with_docker.sh /absolute/path/to/input /absolute/path/to/output
```

This script:

- builds the Docker image
- removes the previous run container before starting a new one
- starts a fresh container for the new run
- mounts the input directory read-only
- writes Markdown, logs, and reports to the output directory

## Docker Validation Scripts

Full Docker-only test run:

```bash
./run_tests_with_docker.sh
```

This script:

- builds a fresh image
- removes the previous Docker test container
- starts a new Docker test container
- runs `python -m doclingflow --help`
- runs the full `unittest` suite inside Docker

Representative mixed-format conversion regression:

```bash
./run_representative_with_docker.sh
```

This script prepares a smaller mixed input set and then calls the normal `run_with_docker.sh` flow.

Heavy PDF validation run:

```bash
./run_heavy_pdf_with_docker.sh
```

This script isolates the expensive PDF paths:

- image-heavy PDFs
- scan-heavy PDFs
- long PDFs
- two-column PDFs

Installability check:

```bash
./run_install_check_with_docker.sh
```

This script starts a fresh Python container, installs the project from the mounted source tree, and then runs `doclingflow --help`.

Build artifact check:

```bash
./run_build_check_with_docker.sh
```

This script starts a fresh Python container and verifies that the project can build distribution artifacts.

Package metadata validation:

```bash
./run_twine_check_with_docker.sh
```

This script builds fresh distribution artifacts in Docker and runs `twine check` against them.

Built-wheel install validation:

```bash
./run_dist_install_check_with_docker.sh
```

This script builds a fresh wheel in Docker and then installs that built artifact into a second fresh Docker container.

## Output Layout

By default, Docker runs write results under `outputs/`:

- `outputs/markdown/`
- `outputs/images/`
- `outputs/reports/`
- `outputs/logs/`

The published Markdown file is the main user-facing output. Intermediate artifacts are kept to support debugging and quality inspection.

## Strategy Model

The pipeline does not treat every document the same.

It distinguishes at least these major routes:

- non-PDF direct conversion
- plain PDF conversion
- scan-heavy PDF conversion
- image-heavy PDF conversion
- two-column PDF conversion
- long PDF conversion with chunking when safe

Key modules:

- `analyzers/file_analyzer.py`
- `analyzers/pdf_analyzer.py`
- `pipeline/strategy_selector.py`
- `pipeline/task_executor.py`
- `pipeline/markdown_pipeline.py`
- `adapters/docling_adapter.py`

## Current Reality Of Heavy PDFs

The project currently succeeds on complex image-heavy PDFs, but those paths are expensive.

In Docker validation, image-heavy long PDFs can take several minutes to finish. That is currently a known cost of the high-fidelity route rather than proof of failure. The project therefore keeps separate Docker validation scripts for:

- faster mixed-format regression
- heavier PDF-specific validation

## Configuration

Runtime settings are still environment-driven and are loaded from `config.py`.

Important variables include:

- `TEST_DOCS_DIR`
- `OUTPUTS_DIR`
- `DEFAULT_MEMORY_LIMIT_MB`
- `CONVERT_TIMEOUT_SEC`
- `PDF_TIMEOUT_SEC`
- `LONG_PDF_TIMEOUT_SEC`
- `SCAN_TIMEOUT_SEC`
- `DOCLING_DOCUMENT_TIMEOUT_SEC`
- `DOCLING_LONG_DOCUMENT_TIMEOUT_SEC`
- `DOCLING_SCAN_DOCUMENT_TIMEOUT_SEC`
- `MARKDOWN_IMAGE_MODE`
- `PDF_SCAN_MODE`
- `LONG_PDF_PAGE_THRESHOLD`
- `LONG_PDF_SIZE_MB_THRESHOLD`
- `PDF_CHUNK_SIZE`
- `PDF_MIN_CHUNK_SIZE`

## Package Direction

The repository is in the middle of a toolization transition.

What is already in place:

- `pyproject.toml`
- `doclingflow` package entrypoint
- CLI subcommands
- Docker-only validation scripts
- installability check inside Docker

What still needs continued work:

- deeper internal implementation migration if the project later wants a stricter package-only layout
- actual publication to PyPI and a public Docker registry
- more targeted optimization of the image-heavy PDF path without sacrificing fidelity

## Documentation

Additional docs:

- `docs/INSTALL.md`
- `docs/CLI.md`
- `docs/DOCKER_WORKFLOW.md`
- `docs/OUTPUTS.md`
- `docs/RELEASE_CHECKLIST.md`
- `docs/STABLE_INTERFACES.md`
- `docs/TOOLIZATION_STAGE_REPORT.md`
- `docs/PUBLISHING.md`

## Positioning

This project should be understood as:

**a Docling-based high-fidelity Markdown converter**

It is not meant to replace the core Docling project. It is meant to provide a stricter, more conservative conversion workflow for users who care about preserving document information for downstream LLM use.
