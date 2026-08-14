# docs/ — design notes and audits

Most documents here are **written in Korean** and predate the current repository layout. The
path mapping is at the bottom of this page.

## Read these first

| | |
|---|---|
| [SETUP.md](SETUP.md) | environment setup · where the data came from · validation procedure |
| **[SCALE_AUDIT.md](SCALE_AUDIT.md)** | ★**required before quoting any absolute number.** Leadfield scale audit |
| [PIPELINE.md](PIPELINE.md) | integrated design: TIP frontend with Sim4Life and NEURON backends |

## Leadfield scale — the longest open problem

How many times too large the absolute field is has been settled and re-opened repeatedly.

| | |
|---|---|
| [LEADFIELD_ORIGIN_CHECK.md](LEADFIELD_ORIGIN_CHECK.md) | tracing the origin of the legacy set |
| [LEADFIELD_REPLY_2.md](LEADFIELD_REPLY_2.md) | follow-up verdict |
| [SCALE_VALIDATION_HANDOFF.md](SCALE_VALIDATION_HANDOFF.md) | validation handoff |

**Current verdict**: the legacy `leadfieldF` reads 2.3–2.5× high per unit current. Three
independent paths agree (single-electrode field, electrode-pair field, M1 over 52 montages).
That is why `rebuild` is the default set.

## NEURON and axon models

| | |
|---|---|
| [NEURON_ROLE.md](NEURON_ROLE.md) | what NEURON must answer and what it must not |
| [NEURON_HYPOTHESES.md](NEURON_HYPOTHESES.md) | hypothesis list and status |
| [MAC_NEURON_HANDOFF.md](MAC_NEURON_HANDOFF.md) · [MAC_FREQ_SWEEP_HANDOFF.md](MAC_FREQ_SWEEP_HANDOFF.md) · [MAC_FETCH_LIST.md](MAC_FETCH_LIST.md) | Mac → WSL port record (port complete) |

## Other

| | |
|---|---|
| [PRACTICE1_GUIDE.md](PRACTICE1_GUIDE.md) | Sim4Life walkthrough |
| [WORKLOG_2026-08-05.md](WORKLOG_2026-08-05.md) | work log |

---

## Old → new paths

| Old | Now |
|---|---|
| `tip/tip/` | `src/tip/` |
| `tip/data/leadfieldF/` | `inputs/leadfield/leadfieldF/` |
| `tip/data/bmask1010.npy` | `inputs/geometry/bmask1010.npy` |
| `tip/data/masks/` | `inputs/masks/mida/` |
| `tip/data/jobs/` | `outputs/jobs/` |
| root `validate_*.py` etc. | `research/<category>/` |
| root `rebuild_solve_batch.py` etc. | `tools/s4l/` |

The current structure is described in [../README.md](../README.md).
