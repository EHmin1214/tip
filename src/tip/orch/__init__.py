# -*- coding: utf-8 -*-
"""orch — the job layer between the TIP frontend and the heavy backends
(Sim4Life and NEURON under WSL).

Design notes in `PIPELINE.md` §3. Two points carry everything else:

- **Jobs outlive the browser.** Solving one electrode takes a couple of minutes and
  regenerating a leadfield takes hours. The in-memory `JOBS` dict in `app.py` is lost when
  the server restarts, so `jobs.JobStore` persists them to disk.
- **Backends run as one process per job.** No resident Sim4Life worker: there is a single
  licence seat, and spawning per job gives crash isolation for free (the "no TCP promotion
  needed" conclusion in `PIPELINE.md` D1).
"""
from .jobs import JobStore                     # noqa: F401
from .s4l import solve_electrodes, s4l_python  # noqa: F401
