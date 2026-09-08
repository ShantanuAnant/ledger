"""Score every RelBench RECOMMENDATION task from its run's `-best.pt`, then
merge the MAP cells into `.scratch/board_results.json` for leaderboard.py.

    PYTHONPATH=. .venv/bin/python scripts/board_rec.py --split val

`scripts/rec_ckpts.json` maps "<dataset>/<task>" -> checkpoint path, and may
carry the per-task eval flags the CHECKPOINT does not already record
(`project_through` is an eval-time choice; `retarget` is read from the
checkpoint itself).

Each task runs in its own subprocess, so one failure costs one cell and not
the sweep -- and the FAIL lines are printed, never swallowed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts-json", default="scripts/rec_ckpts.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--only", default="", help="substring filter on <ds>/<task>")
    ap.add_argument("--results", default=str(ROOT / ".scratch"
                                             / "board_results.json"))
    args = ap.parse_args()

    spec = json.loads(pathlib.Path(args.ckpts_json).read_text())
    tmp = tempfile.NamedTemporaryFile("r", suffix=".jsonl", delete=False)
    tmp.close()

    ok, fail = [], []
    for key, cfg in spec.items():
        if args.only and args.only not in key:
            continue
        ds, task = key.split("/")
        ckpt = cfg["ckpt"] if isinstance(cfg, dict) else cfg
        if not pathlib.Path(ckpt).exists():
            fail.append((key, f"missing checkpoint {ckpt}"))
            print(f"FAIL {key}: missing checkpoint {ckpt}", flush=True)
            continue
        cmd = [".venv/bin/python", "scripts/eval_rec.py", "--ckpt", ckpt,
               "--dataset", ds, "--task", task, "--split", args.split,
               "--batch_size", str(args.batch_size), "--json-out", tmp.name]
        if args.split == "test":
            cmd.append("--rebuild-at-test")
        if isinstance(cfg, dict):
            if cfg.get("project_through"):
                cmd += ["--project-through", cfg["project_through"]]
            if cfg.get("refresh"):
                cmd.append("--refresh")
                if cfg.get("refresh_entity"):
                    cmd += ["--refresh-entity", cfg["refresh_entity"]]
        print(f"\n=== {key}  [{args.split}]\n$ {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT,
                           env={**__import__("os").environ, "PYTHONPATH": "."})
        (ok if r.returncode == 0 else fail).append(key)
        if r.returncode:
            print(f"FAIL {key}: exit {r.returncode}", flush=True)

    cells = {}
    for line in open(tmp.name):
        r = json.loads(line)
        cells[f"{r['dataset']} {r['task']}"] = r["metrics"]["link_prediction_map"]

    out = pathlib.Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(cells)
    out.write_text(json.dumps(merged, indent=1, sort_keys=True))

    print(f"\n{len(ok)} ok, {len(fail)} failed -> {out}")
    for f in fail:
        print("  FAILED:", f)
    for k, v in sorted(cells.items()):
        print(f"  {k:<40} MAP {v:8.4f}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
