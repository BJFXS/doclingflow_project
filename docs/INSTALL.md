# Install Guide

This project is intended to run in Docker first.

## Recommended Path

Use the bundled Docker runtime:

```bash
./run_with_docker.sh
```

Or provide custom paths:

```bash
./run_with_docker.sh /absolute/path/to/input /absolute/path/to/output
```

## Package-Style Entry Point

The repository now exposes the `doclingflow` CLI.

Examples:

```bash
python -m doclingflow --help
doclingflow --help
```

## Docker Installability Check

To verify that the project can be installed as a package:

```bash
./run_install_check_with_docker.sh
```

This runs in a fresh Python container and checks:

- package installation
- console entry point exposure
- `doclingflow --help`

## Build Artifact Check

To verify that the project can build a source distribution and wheel:

```bash
./run_build_check_with_docker.sh
```
