# Release Checklist

This file tracks the minimum steps required before treating the project as a publishable tool.

## Packaging

- `pyproject.toml` is present
- package version is defined
- console script entrypoint is defined
- package URLs point to this repository
- Docker install check passes
- Docker build check passes
- Docker `twine check` passes
- Docker built-wheel install check passes

## Runtime

- `run_with_docker.sh` uses the package CLI entrypoint
- `Dockerfile` default command uses the package CLI entrypoint
- `docker-compose.yml` uses the package CLI entrypoint

## Validation

- Docker-only unit test run passes
- representative mixed-format conversion run passes
- heavy PDF validation is tracked separately

## Documentation

- README explains installation direction
- README explains Docker runtime path
- README explains test scripts
- README explains current heavy-PDF caveat
- `docs/PUBLISHING.md` explains release steps
- `docs/STABLE_INTERFACES.md` defines the supported surface
