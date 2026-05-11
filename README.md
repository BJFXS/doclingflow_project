# doclingflow_project

doclingflow_project is a Docker-first document-to-Markdown pipeline built on top of [Docling](https://github.com/docling-project/docling).

The project is designed for one goal: preserve as much source information as possible while converting common document formats into Markdown that is easier for LLM workflows to consume.

It does not treat all documents the same. Instead, it analyzes each file, routes it through a content-aware strategy, and applies targeted post-processing for structure repair, OCR fallback, image handling, and Markdown cleanup.

## What This Project Does

- Converts common document types to Markdown.
- Uses Docling as the primary conversion backend.
- Applies different strategies for:
  - non-PDF documents
  - short PDFs
  - long PDFs
  - scan-heavy PDFs
  - image-heavy PDFs
  - two-column PDFs
- Exports Markdown, extracted image assets, per-file benchmark rows, summary reports, and logs.
- Runs inside Docker by default so the runtime is reproducible.

## Supported Input Formats

The repository currently scans and converts these file types:

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

The suffix rules are centralized in `document_types.py`.

## How It Works

The default runtime path is:

1. `run_with_docker.sh`
2. `Dockerfile`
3. `main.py`
4. `pipeline/batch_runner.py`
5. `pipeline/task_executor.py`
6. `pipeline/markdown_pipeline.py`
7. `adapters/docling_adapter.py`

At a high level:

1. Collect files recursively from the input directory.
2. Analyze file type and PDF characteristics.
3. Select a processing strategy.
4. Run Docling with strategy-specific runtime options.
5. Repair and normalize the Markdown output.
6. Export reports and logs.

## Strategy Overview

Strategy selection is implemented in `pipeline/strategy_selector.py`.

### Non-PDF

Non-PDF files are routed through `not_pdf` mode.

Typical behavior:

- direct Docling conversion
- lightweight Markdown cleanup
- format-aware image export for HTML, Office, presentation, and image inputs

### Short PDF

Short PDFs are routed through `pdf_short` mode and then classified into content types such as:

- `pdf_plain`
- `pdf_scan`
- `pdf_image`
- `pdf_two_column`

These types change OCR, batching, image export, and post-processing behavior.

### Long PDF

Long PDFs are routed through `pdf_long` mode.

If the document can be chunked safely, the pipeline uses page-range chunking. Otherwise it keeps the document whole and widens timeouts while using more conservative runtime settings.

### OCR and Quality Recovery

The runtime includes PDF quality guards and OCR-aware recovery steps, including:

- unreadable text-layer detection
- source-page based recovery when appropriate
- OCR retry strategies for selected cases
- conservative handling for scan/OCR-heavy outputs

Relevant modules:

- `pipeline/pdf_quality.py`
- `pipeline/ocr_quality.py`
- `pipeline/retry_recovery.py`

## Docker Usage

This project is intended to run in Docker.

### Default Run

Run the bundled test set:

```bash
./run_with_docker.sh
```

By default this script:

- builds the image `doclingflow2:latest`
- uses `test_docs/` as input
- writes results to `outputs/`
- records the last container name in `outputs/reports/.last_container`
- removes the previous run container before starting a new one

### Custom Input and Output Directories

```bash
./run_with_docker.sh /absolute/path/to/input /absolute/path/to/output
```

### Manual Docker Run

```bash
docker build -t doclingflow2:latest .

docker run --rm \
  -e TEST_DOCS_DIR=/data/input \
  -e OUTPUTS_DIR=/data/output \
  -e DEFAULT_MEMORY_LIMIT_MB=12288 \
  -v /absolute/path/to/input:/data/input:ro \
  -v /absolute/path/to/output:/data/output \
  doclingflow2:latest
```

### Docker Compose

A `docker-compose.yml` file exists as an alternate entrypoint, but the normal project path is still `run_with_docker.sh`.

## Output Layout

By default, generated files are written under `outputs/`:

- `outputs/markdown/`
  - published Markdown results
  - per-document artifact directories such as `document.md`, chunk outputs, and exported images
- `outputs/images/`
  - reserved image output root
- `outputs/reports/`
  - latest CSV and summary JSON
  - timestamped benchmark history
- `outputs/logs/`
  - latest run log
  - timestamped run logs

The published Markdown file is the top-level result consumed by reports. Artifact Markdown such as `.../<stem>/document.md` is kept for debugging and intermediate inspection.

## Configuration

Runtime settings are loaded from environment variables in `config.py`.

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

## Image Handling

Image handling is implemented in `processors/image_handler.py`.

Current behavior includes:

- normalizing Markdown image references
- replacing `<!-- image -->` placeholders with real Markdown image links when exported image files exist
- rewriting references to usable relative paths
- preserving scan-specific image evidence blocks for OCR-heavy documents

## Markdown Repair and Cleanup

Post-processing is split into dedicated modules:

- `processors/structure_repair.py`
- `processors/special_block_handler.py`
- `processors/markdown_cleaner.py`
- `processors/formula_detection.py`
- `processors/math_utils.py`

These stages are used to improve structural fidelity, especially for PDFs with OCR, formulas, complex blocks, and layout damage.

## Reports and Benchmark Rows

Each run produces:

- a per-file CSV report
- a summary JSON
- a full run log

The benchmark metrics logic lives in `benchmarks/benchmark_metrics.py`, and row assembly is handled by `pipeline/result_collector.py`.

Metrics include document-level properties such as:

- strategy mode
- content type
- success/failure
- elapsed time
- memory usage
- output page count
- Markdown statistics
- OCR-related annotations
- image reference counts

## Repository Structure

```text
doclingflow_project/
├── adapters/          # Docling adapter layer
├── analyzers/         # file profiling and PDF analysis
├── benchmarks/        # metrics and report row helpers
├── pipeline/          # strategy selection, execution, recovery, result collection
├── processors/        # markdown/image/formula/block post-processing
├── scripts/           # helper scripts used outside the main runtime
├── tests/             # unit tests
├── test_docs/         # bundled sample input set
├── config.py          # environment-backed settings
├── document_types.py  # supported suffixes and content-type helpers
├── Dockerfile
├── run_with_docker.sh
└── main.py
```

## Development Notes

- The runtime is Docker-first.
- The main conversion backend is Docling.
- The project favors conversion completeness and information preservation over raw speed.
- Many outputs keep both published Markdown and intermediate artifacts to make debugging and regression analysis easier.

## Dependencies

Core Python dependencies are listed in `requirements.txt`:

- `docling==2.91.0`
- `pypdf==5.5.0`
- `torch==2.2.2`
- `torchvision==0.17.2`
- `transformers>=4.57,<5`
- `numpy==1.26.4`

The Docker image also installs system packages required for conversion and OCR, including LibreOffice, Poppler, and Tesseract.
