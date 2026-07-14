# Contributing

Thanks for helping improve PQuantML. Keep changes focused and easy to review.

## Workflow

1. Open or pick an issue that describes the bug, feature, or documentation change.
2. Create a branch from the current development branch `dev` using a short descriptive name with issue number
    - (ex. for `#42` issue number `42-tooling`)
3. Make the change with tests or documentation updates that match the scope.
4. Run the relevant local checks before opening a pull request (using existing pre-commit pipeline).
5. Open a pull request that links the issue and summarizes the behavior change.
6. Address review feedback with follow-up commits. A maintainer will merge after approval and passing checks.

## Development setup

Install the project in editable mode with the development extra:

```bash
pip install -e ".[dev]"
```

Run the focused tests for your change, or the full test when changing shared behaviour:

```bash
pytest
```
