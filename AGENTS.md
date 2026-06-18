# Codex Repository Instructions

## Role

This checkout is the local coding workspace for `timetensors`. Codex should help
edit code, documentation, tests, and Slurm scripts, but should not treat this
machine as the experiment runner. Full Slurm experiments are launched on the
distant cluster from another work computer.

The repository owner handles git commits, pushes, branches, and pull requests.
Codex may inspect git state when useful, but should not commit or push unless
explicitly asked.

## Local File Roles

`timetensors/` is the current package. Main entrypoints are `experiment.py`,
`load_dataset.py`, `train_model.py`, and `eval_model.py`. Dataset code lives in
`timetensors/dataset/`, model/loss/normalization code in `timetensors/models/`,
visualization helpers in `timetensors/visu/`, and Slurm scripts in
`timetensors/slurm/`.

`README.md` describes the package: configs, data workflow, outputs, and Slurm
interface. Keep Codex workflow notes out of it.

`timetensors_old/` contains previous implementations, including pipelines and
features not reimplemented in the current package. Use it only for occasional
comparison when debugging or porting behavior; do not import from it in new
package code.

`tests/` contains lightweight local smoke-test scripts for configs,
dataloaders, and models. These scripts use tiny synthetic data and must not
launch Slurm jobs.

`script_outputs/`, datasets, model checkpoints, and experiment outputs are
runtime artifacts. Do not commit them unless the owner explicitly asks.

## Local Tools

The Codex PowerShell PATH may fail to resolve installed tools directly, and the
sandbox may deny running user-installed binaries from AppData. If `python`,
`git`, or `pdflatex` appears missing, use these verified full paths and request
unsandboxed execution when needed:

```powershell
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' --version
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' -c "import torch; print(torch.__version__)"
& 'C:\Program Files\Git\cmd\git.exe' --version
& 'C:\Program Files\Git\bin\bash.exe' --version
& 'C:\Users\Gaspard\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe' --version
```

Verified locally: Python `3.13.14`, global Torch `2.12.0+cpu`, Git
`2.45.0.windows.1`, and MiKTeX-pdfTeX `4.23`. The project itself pins
`torch==2.5.1`; use the project or cluster environment for CUDA-sensitive
checks and do not infer cluster CUDA behavior from the local CPU Torch install.

## Development Commands

Run component checks from the repository root:

```powershell
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' tests\test_config_defaults.py
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' tests\test_dataloaders.py
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' tests\test_models.py
```

Check syntax on touched files before handoff:

```powershell
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' -m py_compile <files>
```

Slurm scripts are cluster experiment launchers. Keep them synchronized with
pipeline defaults and config names, but do not submit them from this local
machine. Syntax-check with Git Bash when needed:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' -n timetensors/slurm/benchmark_models.slurm
```

## Coding Conventions

Use Python 3.10+ syntax, 4-space indentation, and focused changes that match the
existing module patterns. Prefer canonical config names only: model keys such as
`dlinear`, `patchtst`, `chronos`, `tabpfn`; losses `mse`, `nmse`, `mae`,
`nmae`, `rmse`; and normalization names documented in `README.md`.

Keep logging concise and route experiment-level progress through
`experiment.py`. Keep README package-facing, and put Codex/local workflow
instructions here.
