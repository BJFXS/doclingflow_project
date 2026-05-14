# Output Layout

By default, conversion runs write artifacts under `outputs/`.

## Main Directories

- `outputs/markdown/`
- `outputs/images/`
- `outputs/reports/`
- `outputs/logs/`

## Markdown Output

The published Markdown files under `outputs/markdown/` are the main user-facing results.

The repository may also keep intermediate artifacts for debugging and inspection, including:

- per-document working directories
- chunk directories for long PDFs
- exported images
- intermediate `document.md` files

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
