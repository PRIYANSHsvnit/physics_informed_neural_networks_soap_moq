# PINN + SOAP/MoQ Research — Stage 1

## Goal
Stage 1 is **baseline/reproducibility work only**. We are not claiming a new method yet.

The sequence is:
1. Verify a PINN implementation on a controlled 2-D Poisson problem with Adam.
2. Add the official SOAP implementation from the SOAP repository.
3. Verify that SOAP runs and compare Adam vs SOAP under the same PINN/training setup.
4. Only after this, implement the ICLR-2026 paper's SOAP+MoQ equation and reproduce the fixed-μ behavior.
5. Record gradient diagnostics needed for the later reliability-aware study.

## Why Poisson first?
It is deterministic, has a closed-form solution, and is much easier to debug than Burgers or Kovasznay flow. This is a **sanity benchmark**, not a claim of reproducing the exact ICLR-2026 table.

## Experiment discipline
- Do not tune optimizers until all baselines use the same model, seed, collocation points, and stopping budget.
- Record the exact git commit/version of every external optimizer implementation.
- Keep raw JSON/CSV outputs; never overwrite a completed run.
- Report mean/std across seeds only after the single-run pipeline is verified.

## External references
- Original SOAP: https://github.com/nikhilvyas/SOAP
- PINN/SOAP benchmark code: https://github.com/PredictiveIntelligenceLab/jaxpi (branch `pirate` contains the code associated with the 2025 gradient-alignment paper).
- Target ICLR-2026 paper: "Momentum-Accelerated Structured Preconditioning for Physics-Informed Neural Networks".

## Local setup (PowerShell)
Use a dedicated environment. Python 3.12 is recommended for reproducibility, but a recent PyTorch + Python 3.13 environment may also work.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy matplotlib pandas pyyaml
```

Then obtain the official SOAP source:

```powershell
git clone https://github.com/nikhilvyas/SOAP.git external/SOAP
Copy-Item external/SOAP/soap.py src/soap_official.py
```

If Git is unavailable, download `soap.py` from the official repository manually and place it at `src/soap_official.py`.

## First run

From this directory:

```powershell
python src/stage1_poisson.py --optimizer adam --steps 2000 --seed 0 --K 1 --out results/adam_seed0_K1.json
```

After the official SOAP file is present:

```powershell
python src/stage1_poisson.py --optimizer soap --steps 2000 --seed 0 --K 1 --out results/soap_seed0_K1.json
```

These are **smoke tests**. Do not compare them to the paper's reported numbers.

## Stage-1 acceptance criteria
A baseline is considered working only when:
- loss decreases without NaNs/Infs;
- boundary loss remains numerically controlled;
- relative L2 is finite and improves during training;
- parameter/gradient tensor shapes are stable;
- repeated run with the same seed is reproducible to a small numerical tolerance;
- Adam and SOAP use exactly the same data/model/training loop.

## Research log
Use `notes/research_log.md` to record every implementation choice, software version, command, result, and anomaly.
