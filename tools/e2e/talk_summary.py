"""Talk calls of one e2e run: utterance, status, latency, the action taken, plan_revision before/after,
then every change of (plan revision, step, plan-card states) seen by the 100 ms sampler.

Usage: python3 talk_summary.py runs/<name>/timeline.json
"""

import json
import sys

ACTIONS = ("step_say", "target", "step_mark", "go_to", "replan")


def main(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    clip0 = data["clip0"]
    for rec in data["net"]:
        if not rec["url"].endswith("/api/guide/talk"):
            continue
        req, res = rec.get("req") or {}, rec.get("res") or {}
        action = next((name for name in ACTIONS if res.get(name) is not None), "none" if "reply" in res else "-")
        print(json.dumps({
            "clipMs": round(rec["t0"] - clip0),
            "utterance": req.get("utterance"),
            "status": rec.get("status"),
            "blocked": rec.get("blocked"),
            "latencyMs": round(rec["tEnd"] - rec["t0"]) if rec.get("tEnd") is not None else None,
            "action": action,
            "value": res.get(action) if action in ACTIONS else None,
            "reply": res.get("reply"),
            "planRevision": [req.get("plan_revision"), res.get("plan_revision")],
            "error": res.get("error"),
            "reason": res.get("reason"),
        }, ensure_ascii=False))
    last = None
    for sample in data["samples"]:
        key = (sample.get("planRev"), sample.get("stepId"), sample.get("steps"))
        if key != last:
            print(f"clip {sample.get('clipT')} ms  rev={key[0]} step={key[1]} {key[2]}")
            last = key


if __name__ == "__main__":
    main(sys.argv[1])
