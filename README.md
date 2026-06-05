# doclingflow_project

`doclingflow_project` is a document-to-Markdown tool built on top of [Docling](https://github.com/docling-project/docling).

Its goal is to convert common document types such as `pdf`, `pptx`, `docx`, `xlsx`, and `html` into Markdown that is easier to use in LLM workflows such as RAG, indexing, review, and downstream automation.

This project is not a new parser from scratch. It is a Docling-based conversion layer that makes document conversion easier to run end to end:

- one Docker-first workflow
- one CLI for common document types
- strategy-aware PDF routing
- scan-aware OCR handling, including Chinese scan OCR support
- Markdown, logs, and batch reports in one run

## Run It In Docker

This project should be run in Docker.

That is the intended user path, the tested path, and the supported path for normal use.

You can try to run parts of it without Docker, but that is not the recommended workflow and it may fail because of local environment issues such as:

- missing system packages
- OCR/runtime dependency mismatches
- model/runtime version drift
- different local Python, Torch, or Docling environments

If you want the most stable behavior and the lowest setup risk, use Docker.

The project is optimized for a stricter target than simple one-off conversion:

- convert common document formats into Markdown for LLM workflows
- preserve as much source information as possible
- apply strategy-aware handling for difficult PDFs instead of treating all files the same

## Why Docling

This project uses Docling as its base because Docling is a strong foundation for the kind of workflow this repository is trying to provide:

- one conversion base across common office and web document types
- strong PDF-to-Markdown structure extraction compared with lighter Markdown-only tools
- built-in OCR and pipeline options that can be extended with project-specific routing
- a good fit for Dockerized batch workflows instead of ad hoc local-only conversion

This is not a claim that other converters are bad. It is a statement about fit for this repository's goals.

Compared with `MarkItDown`, Docling is a better base when the priority is not just quick Markdown output, but also PDF structure retention, OCR integration, and room for strategy-aware routing.

Compared with `MinerU`, the current project direction favors a single Docling-based workflow that is easier to package, test, and run through one Docker path. MinerU remains a relevant reference point, but it is not the base chosen for this repository today.

## Why Use This Project Instead Of Plain Docling

If you only need to convert a simple file once and you are already comfortable setting up Docling yourself, plain Docling may be enough.

This repository exists for the cases where users want a more complete and easier-to-run workflow:

- one Docker-first path instead of manual local environment setup
- one entry point for common document types instead of separate ad hoc commands
- PDF strategy routing for `plain`, `image-heavy`, `scan-heavy`, `two-column`, and `long` PDFs
- scan-aware OCR handling with Chinese scan OCR support
- batch logs and CSV / JSON reports so bad results are visible instead of silent

In short, Docling provides the core conversion engine. This project packages that engine into a more opinionated workflow so users do not need to keep tuning environment details, OCR setup, or PDF handling by hand.

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
- PDF route selection for `plain`, `image-heavy`, `scan-heavy`, `two-column`, and `long` PDFs
- long PDF chunking
- scan-aware OCR handling with Chinese OCR support
- Markdown structure repair
- image reference normalization
- CSV / JSON batch reports

## Recommended User Path

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

## CLI Entry Point

The project exposes a package-style CLI, but normal users should think of this as the command that runs inside Docker:

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

## Docker Runtime

This repository is intended to run in Docker.

The normal end-to-end path is:

1. `run_with_docker.sh`
2. `Dockerfile`
3. `python -m doclingflow batch /data/input -o /data/output`

## Non-Docker Note

There is a package/CLI structure in this repository, and commands such as `doclingflow --help` or `python -m doclingflow --help` may work in some environments.

That does not change the recommended usage policy:

- normal users should run the project in Docker
- non-Docker runs are secondary and may require manual environment repair

## Docker Validation Scripts

Full Docker-only test run:

```bash
./run_tests_with_docker.sh
```

This script:

- rebuilds the project image
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
- `outputs/artifacts/`
- `outputs/reports/`
- `outputs/logs/`

The published Markdown files under `outputs/markdown/` are the main user-facing outputs.

In the current implementation:

- final Markdown stays under `outputs/markdown/`
- image files referenced by that final Markdown are usually written under per-document `document_artifacts/` directories inside `outputs/markdown/`
- intermediate `document.md` files and similar debugging artifacts are moved under `outputs/artifacts/`
- `outputs/images/` is still part of the configured output layout, but many normal runs will leave it empty

## Strategy Model

The pipeline does not treat every document the same.

It first analyzes the input and then chooses a route instead of sending every PDF through the same settings.

It distinguishes at least these major routes:

- non-PDF direct conversion
- plain PDF conversion
- scan-heavy PDF conversion
- image-heavy PDF conversion
- two-column PDF conversion
- long PDF conversion with chunking when safe

The scan route now supports Chinese OCR in the Docker workflow, which matters for scanned Chinese PDFs that would otherwise degrade into low-quality text output.

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
- `LONG_PDF_CHUNK_TIMEOUT_SEC`
- `LONG_PDF_CHUNK_TIMEOUT_BUFFER_SEC`
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

The repository already has a working toolized path for normal use.

What is already in place:

- `pyproject.toml`
- `doclingflow` package entrypoint
- CLI subcommands
- Docker-only validation scripts
- installability check inside Docker
- Dockerized OCR support for scan-heavy PDFs, including Chinese scan OCR support

What still needs continued work:

- deeper internal implementation migration if the project later wants a stricter package-only layout
- actual publication to PyPI and a public Docker registry
- more targeted optimization of the image-heavy PDF path without sacrificing fidelity

## Best Fit

This project is a good fit when you want to:

- convert common documents into Markdown for LLM workflows
- avoid hand-fixing OCR dependencies and local runtime drift
- batch process real-world PDFs instead of only simple clean files
- keep logs and reports for conversion review

It is less important when you only want to run a one-off conversion on a simple file and are already comfortable using plain Docling directly.

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

It is not meant to replace the core Docling project. It is meant to provide a stricter, Docker-first, easier-to-run conversion workflow for users who want common document types converted to Markdown for downstream LLM use without hand-managing OCR setup, environment details, and PDF routing logic.
