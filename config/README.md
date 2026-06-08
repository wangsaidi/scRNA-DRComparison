# Configuration and Environment Records

This directory stores environment and provenance records used for benchmark assembly and targeted sensitivity analyses.

## Contents

- `benchmark_environments/`: environment exports, package-freeze files, method-install records and source-commit records.

## Interpretation

The benchmark spans methods implemented across Python, R, Octave and older software environments. A single universal environment is therefore not expected. Use these files to identify the environment associated with a given method or rerun script.

Before public release, the authors should confirm whether a simplified container, conda-lock file or archival software snapshot will be provided in addition to these environment records.
