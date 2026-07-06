# Output Layout

By default, conversion runs write artifacts under `outputs/`.

## Main Directories

- `outputs/*.md`
- `outputs/*.assets/`
- `outputs/artifacts/`
- `outputs/reports/`
- `outputs/logs/`

## Markdown Output

The published Markdown files directly under `outputs/` are the main user-facing results.

Each published Markdown file can have a sibling asset directory:

- `outputs/report_001.md`
- `outputs/report_001.assets/`

Image references in the published Markdown are rewritten to point into that sibling asset directory, so collecting final deliverables does not require drilling into per-document working folders.

The repository keeps intermediate artifacts for debugging and inspection under `outputs/artifacts/`, including:

- per-document working directories
- chunk directories for long PDFs
- intermediate `document.md` files

The repository may also use `outputs/markdown/` as an internal staging area while building the final published Markdown.

## Reports

`outputs/reports/` contains:

- rolling latest report files
- timestamped benchmark CSV files
- timestamped summary JSON files

## Logs

`outputs/logs/` contains:

- a rolling latest log
- timestamped run logs

For heavy PDFs, logs are especially important because they show:

- monitor samples
- chunk counts
- chunk-level progress
- timeout or hang detection
