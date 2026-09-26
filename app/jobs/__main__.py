"""
Run scheduled jobs from the command line (any scheduler; AWS EventBridge -> ECS in production).

    python -m app.jobs sync_products
    python -m app.jobs refresh_fulfilment --budget-minutes 5
    python -m app.jobs refresh_instagram_tokens --store store_abc     # one store, ignores the interval
    python -m app.jobs all

Exit codes: 0 succeeded / skipped (already running) · 1 partial · 2 failed · 3 unknown job
"""

import argparse
import json
import sys
from datetime import timedelta

from app.jobs.runner import run_job
from app.jobs.tasks import JOBS

_EXIT = {"succeeded": 0, "skipped_locked": 0, "partial": 1, "failed": 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.jobs")
    parser.add_argument("job", help=f"one of: {', '.join(JOBS)}, all")
    parser.add_argument("--budget-minutes", type=float, default=10.0)
    parser.add_argument("--store", default=None)
    args = parser.parse_args(argv)

    names = list(JOBS) if args.job == "all" else [args.job]
    if any(n not in JOBS for n in names):
        print(f"Unknown job '{args.job}'. Available: {', '.join(JOBS)}, all", file=sys.stderr)
        return 3

    worst = 0
    for name in names:
        s = run_job(JOBS[name], budget=timedelta(minutes=args.budget_minutes), only_store=args.store)
        print(json.dumps({"job": s.job, "status": s.status, "processed": s.processed, "skipped": s.skipped,
                          "failed": s.failed, "stopped_early": s.stopped_early, "stores": s.stores}, default=str))
        worst = max(worst, _EXIT[s.status])
    return worst


if __name__ == "__main__":
    sys.exit(main())