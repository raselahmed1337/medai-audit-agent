"""Experimental comparison runner.

Factorial: {A1, A2, A3} x failure_rate {0.0, 0.2, 0.4} x 12 tasks x R seeds.
Writes results/results.json and results/table.md (means with 95% bootstrap CIs).
"""
from __future__ import annotations

import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.architectures import ARCHITECTURES  # noqa: E402
from eval.metrics import (citation_metrics, reliability_metrics, safety_metrics,  # noqa: E402
                          task_success)
from eval.tasks import TASKS, gold_answer  # noqa: E402

FAILURE_RATES = [0.0, 0.2, 0.4]
REPEATS = 5  # seeds per (arch, rate, task)


def run_all(repeats: int = REPEATS, out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for arch_name, runner in ARCHITECTURES.items():
        for rate in FAILURE_RATES:
            for task in TASKS:
                gold = gold_answer(task)
                for rep in range(repeats):
                    seed = zlib.crc32(f"{arch_name}|{rate}|{task['task_id']}|{rep}".encode())
                    try:
                        result = runner(task["query"], rate, seed)
                    except Exception as e:
                        from experiments.architectures import RunResult
                        result = RunResult(architecture=arch_name, query=task["query"],
                                           status=f"failed:unhandled:{e!r}")
                    cm = citation_metrics(result.citations, gold["gold_citations"])
                    row = dict(
                        architecture=arch_name, failure_rate=rate, task=task["task_id"],
                        rep=rep, status=result.status, success=task_success(result, gold),
                        citation_f1=cm["citation_f1"],
                        citation_precision=cm["citation_precision"],
                        citation_recall=cm["citation_recall"],
                        n_fabricated=cm["n_fabricated"],
                        analysis_ran=result.analysis_ran,
                        **safety_metrics(result),
                        audit_ok=result.audit_ok,
                    )
                    rows.append(row)
    # aggregate
    table = {}
    for arch in ARCHITECTURES:
        for rate in FAILURE_RATES:
            sub = [r for r in rows if r["architecture"] == arch and r["failure_rate"] == rate]
            key = f"{arch}@p={rate}"
            table[key] = _bootstrap(sub)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"rows": rows, "summary": table}, f, indent=2)
    _write_md(table, os.path.join(out_dir, "table.md"))
    return rows, table


def _metric_fns(sub):
    return {
        "success_rate": lambda rs: sum(r["success"] for r in rs) / len(rs),
        "citation_f1": lambda rs: sum(r["citation_f1"] for r in rs) / len(rs),
        "fabrication_rate": lambda rs: sum(r["n_fabricated"] > 0 for r in rs) / len(rs),
        "approval_violation_rate": lambda rs: sum(r["approval_violation"] for r in rs) / len(rs),
        "audit_intact_rate": lambda rs: sum(r["audit_ok"] for r in rs) / len(rs),
    }


def _bootstrap(sub, n_boot: int = 2000, seed: int = 7):
    rng = random.Random(seed)
    fns = _metric_fns(sub)
    out = {}
    for name, fn in fns.items():
        point = fn(sub)
        boots = []
        for _ in range(n_boot):
            sample = [sub[rng.randrange(len(sub))] for _ in sub]
            boots.append(fn(sample))
        boots.sort()
        lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]
        out[name] = dict(mean=round(point, 4), ci95=[round(lo, 4), round(hi, 4)])
    return out


def _write_md(table, path):
    cols = ["success_rate", "citation_f1", "fabrication_rate",
            "approval_violation_rate", "audit_intact_rate"]
    lines = ["# Architecture comparison (mean [95% bootstrap CI])", "",
             "| architecture@failure_rate | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for key, m in table.items():
        cells = [f"{m[c]['mean']:.3f} [{m[c]['ci95'][0]:.3f}, {m[c]['ci95'][1]:.3f}]" for c in cols]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else REPEATS
    rows, table = run_all(repeats)
    print(f"{len(rows)} runs complete -> results/results.json, results/table.md")
    for key, m in table.items():
        print(key, {k: v["mean"] for k, v in m.items()})
