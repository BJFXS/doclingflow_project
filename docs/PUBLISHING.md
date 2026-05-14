# Publishing

This document describes the release path for `doclingflow` after the non-PDF toolization work.

## Release Goal

The project should be usable without `git clone`. The preferred release surfaces are:

- PyPI package: `pip install doclingflow`
- Docker image: `docker pull <published-image>`

Source checkout remains a development path, not the preferred end-user path.

## Required Docker Validation Before Release

Run these checks from the repository root:

```bash
./run_tests_with_docker.sh
./run_representative_with_docker.sh
./run_build_check_with_docker.sh
./run_twine_check_with_docker.sh
./run_dist_install_check_with_docker.sh
```

Interpretation:

- `run_tests_with_docker.sh`: package CLI and unit suite still pass
- `run_representative_with_docker.sh`: common document types still convert end to end
- `run_build_check_with_docker.sh`: `sdist` and `wheel` still build
- `run_twine_check_with_docker.sh`: distribution metadata is publishable
- `run_dist_install_check_with_docker.sh`: a built wheel can be installed in a fresh container

Heavy PDF validation remains a separate gate and should still be run before major releases:

```bash
./run_heavy_pdf_with_docker.sh
```

## PyPI Release Flow

1. Choose the release version and update `pyproject.toml`.
2. Run the required Docker validation commands listed above.
3. Create fresh build artifacts.
4. Upload `dist/*` to the target package index.
5. Verify that a fresh environment can install and run `doclingflow --help`.

Example upload commands, after credentials are configured:

```bash
python -m pip install --upgrade build twine
python -m build
twine upload dist/*
```

## Docker Image Release Flow

1. Build the image from the release tag or release commit.
2. Tag it with a versioned tag and a rolling tag if desired.
3. Push to the chosen registry such as Docker Hub or GHCR.
4. Verify that a fresh host can `docker pull` and run the batch CLI path.

## Stability Policy

The following surfaces should stay stable across normal releases:

- `doclingflow` CLI subcommands
- `python -m doclingflow`
- `run_with_docker.sh`
- `run_tests_with_docker.sh`
- `run_representative_with_docker.sh`
- `run_heavy_pdf_with_docker.sh`
- `run_build_check_with_docker.sh`
- `run_twine_check_with_docker.sh`
- `run_dist_install_check_with_docker.sh`

Root-level implementation modules are still compatibility surfaces, not preferred external APIs.
