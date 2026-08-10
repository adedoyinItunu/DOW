Explainable Denial of Wallet Detection

Artefact for the MSc Research Project *Explainable Denial of Wallet Detection:
Applying XAI Techniques to Serverless Traffic Heat Maps*.

Itunu Deborah Adedoyin · x24249700 · MSc Cloud Computing · School of Computing,
National College of Ireland · Supervisor: Dr. Ahmed Makki

 
What this is

A Denial of Wallet attack inflates a serverless bill rather than degrading
service. Its hardest form is a low rate *leech*, which stays inside normal
traffic thresholds. One published approach renders a month of request traffic as
a 24 × 30 image and classifies it with a convolutional network.

This artefact evaluates that approach under controls it has not previously been
given: a transparent linear model running through every analysis, a chance
baseline for attribution agreement, a random occlusion control for fidelity, and
a one parameter volume threshold as a trivial comparison.

Four findings follow from it:

 Instability across data partitions comes from the interaction of a
 globally pooled classifier head with a short training budget, not from
 convolutional detection as such.
 Detection of the low rate leech degrades monotonically in both models, with a
 transition region below roughly half intensity where the local trace of the
 attack disappears.
 Under a shift in generator parameters the legitimate traffic class collapses
 entirely in both models. A per image robust normalisation substantially
 mitigates this while *improving* low rate detection.
 Attribution methods separate the failure group at group level but fail their
 controls on individual samples.

 

 Environment

Python 3.9 on CPU. No GPU is used and no result depends on GPU numerics.

```bash
python3.9 m venv .venv
source .venv/bin/activate
pip install upgrade pip
pip install r requirements.txt
```

Versions are pinned. Reproduction is deterministic under the pinned NumPy
version; a different NumPy major version can change the stream a seed produces.

 

Quick start

```bash
 generate the default dataset
python dow_data.py per class 300 out data.npz

 confirm the generator is deterministic
python dow_data.py seed 0 out /tmp/a.npz && md5sum /tmp/a.npz
python dow_data.py seed 0 out /tmp/b.npz && md5sum /tmp/b.npz same hash

 run the audits
python check_leakage.py
python audit_checkpoints.py
python reconcile_experiments.py

 interactive demo, then open http://127.0.0.1:5000
python demo_app.py
```

 

 Layout

 Core modules

| File | Purpose |
| | |
| `dow_data.py` | Seeded traffic generator, normalisation, partitioning |
| `dow_model.py` | `DoWNetCNN` and the single seeding function |
| `dow_model_variants.py` | Head variants and parameter counts |
| `dow_data_ORIGINAL.py` | Pre revision generator, retained for comparison |

 Experiments

| File | Analysis |
| | |
| `experiment1_baseline.py` | Baseline across partition varying seeds |
| `architecture_controls.py` | Pooled versus positional head |
| `experiment5_multiseed_sweep.py` | Leech intensity sweep, five seeds |
| `evaluate_cross.py` | Cross configuration transfer |
| `experiment4_normalisation.py` | Normalisation intervention |
| `experiment6_xai_failures.py` | Group attribution, single model |
| `xai_failure_multiseed.py` | Group attribution across five models |
| `fixed_threshold_control.py` | One parameter volume threshold |

 Controls and statistics

`xai_controls.py`, `xai_controls_stats.py`, `xai_failure_stats.py`,
`compute_snr.py`, `test_auc_separability.py`, `test_ridge_leech.py`,
`test_normalisation.py`, `paired_test_cnn_vs_lr.py`, `verify_shap_effect.py`

 Verification

| File | Check |
| | |
| `check_leakage.py` | No overlap between train, validation and test |
| `audit_checkpoints.py` | Each checkpoint matches the protocol it claims |
| `reconcile_experiments.py` | Figures quoted more than once agree |
| `verify_sweep_data.py` | The sweep regenerates data at the intended scales |
| `verify_shap_effect.py` | Re derives the complete separation result by hand |

 Figures

`make_exemplar_figure.py`, `make_results_figures.py`, `plot_e1_confusion.py`,
`plot_lr_coefficients.py`, `plot_multiseed_sweep.py`, `plot_roc_separability.py`,
`plot_val_traces.py`

 Data and outputs

`data*.npz` hold raw hourly counts , not normalised values. Normalisation is
applied at load time, so one stored dataset can be read under either scheme
without regeneration.

`out_*.txt` are the saved console output of every reported run, committed so a
reproduction can be compared against the original. `*.json` hold structured
results. `*.pt` are model checkpoints.

 

 Reproducing the reported results

| Result | Command |
| | |
| Stability across partitions | `python experiment1_baseline.py` |
| Architecture control | `python architecture_controls.py arch nogap protocol fixed25` |
| Leech intensity sweep | `python experiment5_multiseed_sweep.py` |
| Cross configuration transfer | `python evaluate_cross.py` |
| Normalisation intervention | `python experiment4_normalisation.py` |
| Group attribution, five seeds | `python xai_failure_multiseed.py k 25` |
| Threshold comparison | `python test_auc_separability.py` |

Generate the datasets first, then run `check_leakage.py`, then the analyses.
Anything consuming a checkpoint needs `experiment1_baseline.py` to have produced
one. A converged ten seed sweep takes roughly ten minutes on two CPU cores.

 

 Generator

A month is a 24 × 30 matrix of invocation counts. The legitimate baseline is
Poisson about a Gaussian diurnal envelope with a nocturnal floor, modulated
weekly. Four classes: `normal`, `linear`, `geometric`, `random`. Attacks are
generated at two intensities, tagged 1 for the low rate leech and 2 for the
high rate flood.

Every parameter is exposed on the command line, which makes the robustness and
threat analyses controlled experiments rather than code variants:

```
 peak hour width amplitude weekend factor weekly shape
 leech scale flood scale linear const seed
```

The generator reconstructs the documented patterns of the Denial of Wallet Test
Simulator (Kelly, 2022, https://github.com/psykodan/DoWTS). The published
simulator was not used: at its most recent commit it declares a module path that
does not resolve, requires a hardcoded database connection at package scope,
comments out its parameterised entry point, and seeds itself from the wall
clock, so a successful build would still not yield reproducible data.

 

 Known limitations

 All evaluation is on generated data. No public dataset of month long
 serverless traffic with compatible labels was identified.
 The detector was not deployed as a cloud service. The EC2 instance provided
 compute only; the deployment architecture in the report is a design.
 Peak memory, deployment package size once the PyTorch runtime is included,
 and per invocation cost were not measured.
 The threat model is static: every attack is generated without reference to the
 defence.
 The transition region is expressed in generator units. Converting it to a
 percentage of a monthly bill requires the baseline amplitude.

 

 Reusable beyond this study

 The generator , reproducible from a seed where the field's reference
 simulator is not.
 The evaluation protocol , which is method rather than software and applies
 to any detector assessed on generated data.
 The one parameter volume threshold , which needs no training, no labelled
 data and no model to maintain.
