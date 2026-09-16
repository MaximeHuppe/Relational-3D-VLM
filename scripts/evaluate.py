#!/usr/bin/env python3
"""Entry point: the full Phase 4 evaluation report. Not implemented yet.

What exists today lives in ``scripts/train_relational_model.py``: a trained
Stage B checkpoint can be scored on any split, with either anchor source, and
the report is stratified by target shape, anchor shape, direction and clause
slot::

    .venv/bin/python scripts/train_relational_model.py --eval-only \
        --checkpoint runs/stage_b_oracle/best.pt --eval-split test \
        --anchor-source predicted --stage-a-checkpoint runs/stage_a/best.pt

What this script is reserved for is the rest of the Phase 4 deliverable: the
four-way baseline sweep, the scored counterfactual battery
(``src/evaluation/counterfactuals.py``) and the qualitative 3D dumps
(``src/evaluation/qualitative.py``).
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    raise SystemExit(
        "the full Phase 4 report is not implemented yet; for per-split Stage B "
        "metrics with either anchor source use:\n"
        "  .venv/bin/python scripts/train_relational_model.py --eval-only "
        "--checkpoint runs/stage_b_oracle/best.pt"
    )


if __name__ == "__main__":
    sys.exit(main())
