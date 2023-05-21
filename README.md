# AIRA
Ambisonics Impulse Response Analyzer
  
---
## 🌱 **Getting started**

1. Download the repository
    ```bash
    git clone https://github.com/nahue-passano/AIRA.git
    cd AIRA
    ```

2. Create and initialize poetry environment
    ```bash
    poetry install
    poetry shell
    ```

    > **Note**: If the environment already exists, run `poetry update` for possible changes in `pyproject.toml`.

3. Install the pre-commit hooks for code formating and linting with `black` and `pylint`.
    ```bash
    pre-commit install
    ```

    > **Note**: If the changes to be commited are reformated, `black` will cancel the commit. You must add again the changes with `git add` and commit again

---