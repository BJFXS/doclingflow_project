# CLI Guide

The project exposes a package-style CLI named `doclingflow`.

## Help

```bash
doclingflow --help
```

## Convert

Convert a single document:

```bash
doclingflow convert input.pdf -o output.md
```

Useful options:

- `--output-dir`
- `--strategy auto|plain|scan|image|two-column`
- `--ocr auto|force|off`
- `--image-mode referenced|embedded`
- `--emit-report`
- `--disable-chunking`
- `--timeout`
- `--memory-limit-mb`
- `--json`

## Batch

Convert a directory tree:

```bash
doclingflow batch /data/input -o /data/output
```

JSON output is supported:

```bash
doclingflow batch /data/input -o /data/output --json
```

## Inspect

Inspect a file without converting it:

```bash
doclingflow inspect input.pdf
doclingflow inspect input.pdf --json
```

This is useful for checking:

- file family
- selected strategy
- chunk count
- OCR enablement

## Doctor

Check runtime prerequisites:

```bash
doclingflow doctor
```

## Version

Print tool, Docling, and Python versions:

```bash
doclingflow version
```
