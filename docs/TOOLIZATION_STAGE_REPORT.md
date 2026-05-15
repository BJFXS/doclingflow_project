# Toolization Stage Report

This document records the current completion state of the non-PDF toolization work.

## Completed In This Stage

- package-style CLI exists
- package-style Python API exists
- Docker runtime entrypoints use the package CLI
- Docker-only test scripts exist
- Docker-only install check exists
- Docker-only build check exists
- Docker-only `twine` metadata check exists
- Docker-only built-wheel install check exists
- README has been rewritten as a tool-oriented document
- focused docs have been added under `docs/`
- stable interface boundaries are now documented
- package-level namespaces now exist for the main internal module groups

## What This Means

The project is no longer just a repository with internal scripts.

It now behaves like a tool project with:

- a CLI surface
- a Python package surface
- Docker-first validation flows
- documented stable entrypoints
- release-ready build and install dry runs

## Still Deferred

- image-heavy and other complex PDF path optimization
- deeper internal implementation migration away from root-level modules
- removal of redundant deep wrapper modules under `doclingflow/*`
- actual external publication to PyPI and a public Docker registry

Those items are intentionally separate from this non-PDF toolization stage.
