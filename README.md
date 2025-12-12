# PSU Capstone Project

## 1. Environment Setup: uv

This project uses the uv package manager for reproducible Python environments.

### a. Install uv
- Windows (PowerShell):
  ```powershell
  irm https://astral.sh/uv/install.ps1 | iex
  ```
- Linux/MacOS:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- Verify:
  ```bash
  uv --version
  ```

### b. Create & Activate Environment
- From the project root:
  ```bash
  uv python install 3.12
  uv python pin 3.12
  uv sync --all-groups
  ```
- uv manages the virtual environment automatically (in `.venv`); commands like `uv run` use it.

### c. Optional: Recreate Environment
- If you need a fresh environment:
  ```bash
  rm -rf .venv        # Unix
  # or on Windows PowerShell:
  Remove-Item -Recurse -Force .venv
  uv python install 3.12
  uv python pin 3.12
  uv sync --all-groups
  ```

---

## 2. Makefile: Installation & Usage

### a. What is Make?
Make automates development tasks using a `Makefile`.

### b. Install Make
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential
  ```
- **MacOS:**
  ```bash
  xcode-select --install
  ```
- **Windows:**
  - Recommended: WSL or Git Bash
  - Or with Chocolatey:
    ```powershell
    choco install make
    ```

### c. Makefile Dependencies
- `uv` (installed above)
- Tooling: `pre-commit`, `isort`, `black`, `flake8`, `pytest` (resolved by `uv sync`)

---

## 3. Using the Makefile

From the project root, run:
```bash
make <target>
```
Common targets:
- `make help`        # List all commands
- `make install`     # Create/refresh env, install hooks
- `make setup-dev`   # Install dev dependencies
- `make format`      # Format code
- `make lint`        # Lint code
- `make test`        # Run tests
- `make clean`       # Remove build/test artifacts
- `make update`      # Update dependencies (uv lock --upgrade + uv sync)
- `make pre-commit`  # Run pre-commit hooks
- `make recreate-venv` # Force rebuild `.venv`

---

## 4. Troubleshooting

| Issue                      | Cause                              | Fix                                                     |
|---------------------------|------------------------------------|---------------------------------------------------------|
| `uv` not found            | uv not installed                   | Install uv (see Section 1a)                             |
| Wrong Python version      | Env not pinned/installed           | `uv python install 3.12 && uv python pin 3.12`          |
| Dependencies outdated     | Lockfile not upgraded              | `uv lock --upgrade && uv sync --all-groups`             |
| Pre-commit fails          | Missing dependencies               | `make install`                                          |
| Using wrong environment   | Stale `.venv` or external Python   | `make recreate-venv`                                    |

---

## 5. Git Workflow: Commit, Hooks, Sync

1. **Stage changes:**
   ```bash
   git add .
   ```
2. **Commit:**
   ```bash
   git commit -m "Your message"
   ```
   - If pre-commit hooks fail, fix issues, re-stage (`git add .`), and re-run `git commit`.
3. **Sync with remote:**
   ```bash
   git pull --rebase
   git push
   ```

---

## 6. Additional Notes
- For more info on Makefile targets, run `make help`.
- `uv run` executes commands inside the managed virtual environment.
- `.venv` is created automatically; avoid modifying it manually unless recreating.

---

## License
Add license info here if applicable.
