# PSU Capstone Project (FORK - millscb)

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
- If you need a fresh environment and already have `.venv`, you can recreate it in one of three ways:

1) Using the Makefile (recommended):
```bash
make recreate-venv
```
- Removes `.venv`, installs/pins Python 3.12, and runs `uv sync --all-groups`.

2) Manually (Unix/macOS):
```bash
rm -rf .venv
uv python install 3.12
uv python pin 3.12
uv sync --all-groups
```

3) Manually (Windows PowerShell):
```powershell
Remove-Item -Recurse -Force .venv
uv python install 3.12
uv python pin 3.12
uv sync --all-groups
```

Notes:
- Use this when `.venv` points to a removed or incompatible Python, or dependency resolution is broken.
- After recreation, run:
```bash
uv lock --upgrade
uv sync --all-groups
```
to ensure dependencies are updated to the latest allowed versions.

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
| Using wrong environment   | Stale `.venv` or external Python   | `make recreate-venv` or remove `.venv` and re-run uv steps |
| SSL certificate errors (Zscaler) | Corporate proxy with custom certs | Run `./scripts/setup-zscaler-certs.sh` (already configured in Makefile) |

### SSL Certificate Issues (Zscaler/Corporate Proxy)

If you encounter SSL certificate errors like `invalid peer certificate: UnknownIssuer` when running `make update` or other `uv` commands, this is typically caused by a corporate proxy (like Zscaler) intercepting HTTPS traffic.

**Quick Fix:**
The Makefile is already configured to use the combined certificate bundle. The certificates are automatically set up in:
- `~/.local/share/ca-certificates/combined-ca-bundle.crt`

**For direct `uv` usage outside of make:**
Run the setup script to configure your shell environment:
```bash
./scripts/setup-zscaler-certs.sh
```

This will:
1. Extract and save Zscaler certificates
2. Create a combined CA bundle with system + Zscaler certs
3. Optionally add environment variables to your shell profile

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

---

## Key test graphs

![Scalar Encoder Overlap vs Distance not periodic](test_images/scalar_encoder_overlap_vs_distance_not_periodic.png)

This figure shows when you have a base encoding of 1 versus 1 through 1000 with a scalar encoder non-periodic. You can see that far numbers being encoded have zero overlap whereas close ones show similarity.

![Scalar Encoder Overlap vs Distance periodic](test_images/scalar_encoder_overlap_vs_distance_periodic.png)

This figure shows when you have a base encoding of 1 versus 1 through 1000 with a scalar encoder periodic. You can see that similarity is created periodically.

![RDSE Overlap vs Distance](test_images/rdse_overlap_vs_distance.png)

This figure shows when you have a base encoding of 1 versus 1 through 1000 with an RDSE. You can see there is similarity noise generated by this hashing method.

![Spatial Pooler vs Distance with Scalar input field (active column overlaps)](test_images/spatial_pooler_active_col_overlap_vs_distance_with_scalar_input_field.png)

This figure shows when a Spatial Pooler is created with a Scalar Encoder input field. The overlap is computed on the active columns. We can see here that some noise is generated by the Spatial Pooler itself where some similarity is created on distant values. That being said, in HTM theory it is acceptable to have around 50% noise before there are issues.

![Spatial Pooler vs Distance with RDSE input field (active column overlaps)](test_images/spatial_pooler_active_col_overlap_vs_distance_with_rdse_input_field.png)

This figure shows when a Spatial Pooler is created with an RDSE input field. The overlap is computed on the active columns. We can see here that the noise is much larger with some peaks even passing the 50% threshold for acceptability. 

![Spatial Pooler column activation frequency distribution with random data zero epoch excluding encoder](test_images/Activation_Frequency_Distribution_with_random_data_zero_epoch_excluding_encoder.png)

This figure shows the activation frequency distribution of the active columns (for example, around 30% of the columns are active 2% of the time). This is at epoch zero which means no training and random data is used to compute with. On top of that we excluded the encoder and activated cells directly. We can see here that about 86% of the columns participated when activating these cells directly and not using an encoder.

![Spatial Pooler column activation frequency distribution with random cells once excluding encoder](test_images/Activation_Frequency_Distribution_with_random_cells_excluding_encoder_spatial_pooler.png)

This figure shows the activation frequency distribution of the active columns. This is at epoch 49 with random cells excluding an encoder. You can see there is starting to be a rough pattern as we have dominant columns being activated at between 10-15%.

![Spatial Pooler column activation frequency distribution with random data once with encoder](test_images/Activation_Frequency_Distribution_with_random_date_including_encoder_spatial_pooler.png)

This figure shows the activation frequency distribution of the active columns. This is at epoch 49 with random values being encoded by an RDSE. You can see when comparing with Figure C-7 that we have a lot more “dead” columns or columns that do not participate. Overall, only about 60% of columns were ever active during the 49 epochs.

![Spatial Pooler column activation frequency distribution with sine wave with scalar encoder input field and periodic false](test_images/Activation_Frequency_Distribution_with_sine_wave_with_scalar_encoder_periodic_false_spatial_pooler.png)

This figure shows the activation frequency distribution of the active columns. This is at epoch 49 with sine wave values being encoded by a Scalar Encoder non-periodic. Almost 80% of the columns were active at one point during this and we can see some dominant columns between 0.20 and 0.25 which indicates the Spatial Pooler is learning the pattern.

![Spatial Pooler column activation frequency distribution with sine wave and RDSE](test_images/Activation_Frequency_Distribution_with_sine_wave_with_rdse_encoder_spatial_pooler.png)

This figure shows the activation frequency distribution of the active columns. This is at epoch 49 with sine wave values being encoded by an RDSE. Almost 90% of the columns are “dead” and we have extremely dominant columns.

![Spatial Pooler noise robustness with RDSE](test_images/spatial_pooler_noise_robustness.png)

This figure shows a noise robustness plot per epoch of a Spatial Pooler. We can see that the more training done with a Spatial Pooler the more robust it becomes to random noise being added to its input field cells.

![Synapse Formation Two Datasets](test_images/synapse_formation_two_datasets.png)

This figure shows continuous learning in the Spatial Pooler. We can see at epoch 0 we have a lot of synapse formation (this is defined as a synapse permanence passing a threshold). The dotted line is a totally different non-overlapping dataset. We can see new unique synapses start to form before the Spatial Pooler has roughly caught on to the sequence of values. We also see that when the Spatial Pooler has fully adjusted to a dataset that new synapses slow down in formation.

---

#Testing performed on HTM core python bindings for comparison

![rdse overlap versus distance](test_images/htm_core_python_bindings/rdse_overlap_vs_distance.png)

![Scalar encoder overlap versus distance periodic](test_images/htm_core_python_bindings/Scalar_encoder_overlap_vs_distance_periodic.png)

![Spatial Pooler active columns overlap vs distance with rdse](test_images/htm_core_python_bindings/sp_overlap_vs_distance_with_rdse.png)

![Spatial Pooler active columns activation frequency distribution with random cells excluding encoder](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_cells_excluding_encoder.png)

![Spatial Pooler active columns activation frequency distribution with random cells excluding encoder with boosting](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_cells_excluding_encoder_with_100_boost.png)

![Spatial Pooler active columns activation frequency distribution with random data zero epoch excluding encoder](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_zero_epoch_excluding_encoder.png)

![Spatial Pooler active columns activation frequency distribution with random data zero epoch scalar encoder](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_zero_epoch_scalar_encoder.png)

![Spatial Pooler active columns activation frequency distribution with random data zero epoch rdse](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_zero_epoch_rdse.png)

![Spatial Pooler active columns activation frequency distribution with random data once with scalar encoder](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_scalar_encoder.png)

![Spatial Pooler active columns activation frequency distribution with random data once with scalar encoder with boosting](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_scalar_encoder_and_boost.png)

![Spatial Pooler active columns activation frequency distribution with random data once with rdse](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_rdse.png)

![Spatial Pooler active columns activation frequency distribution with random data once with rdse and boosting](test_images/htm_core_python_bindings/activation_frequency_distribution_with_random_data_once_with_rdse_and_boost.png)

![Spatial Pooler active columns activation frequency distribution with sine wave, scalar encoder, and periodic false](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_scalar_encoder_periodic_false.png)

![Spatial Pooler active columns activation frequency distribution with sine wave, scalar encoder, and periodic false with boosting](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_scalar_encoder_periodic_false_and_boosting.png)

![Spatial Pooler active columns activation frequency distribution with sine wave and rdse](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_rdse.png)

![Spatial Pooler active columns activation frequency distribution with sine wave and rdse with boosting](test_images/htm_core_python_bindings/activation_frequency_distribution_with_sin_wave_with_rdse_and_boosting.png)
