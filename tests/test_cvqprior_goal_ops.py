#!/usr/bin/env python3
"""
tests/test_cvqprior_goal_ops.py - goals on cvq_prior_reset / cvq_prior_next of a running `neon --interactive`.

    g++ -std=c++20 -O2 -pthread -mavx2 -mfma -mf16c runtime/NEON-3.cpp -o neon
    python3 tests/test_cvqprior_goal_ops.py --neon ./neon          # needs numpy + gguf (pip install gguf); no torch, no VAE, no real weights

A small random goal-conditioned prior is written with conversion/nanity_export_cvqprior.py (the weights come from the synthetic builder of tests/verify_cvqprior.py), loaded with
--motion-model, and driven through the JSON ops. Random weights are enough: what is checked is which goal reaches which window, not what the motion looks like.

  1 info          the prior reports its goal numbers (names, mean, std)
  2 plan          cvq_prior_reset with "goals" / "goal" gives exactly the codes of the same session whose cvq_prior_next calls carry the goal themselves (seeds, plan lengths
                  1..3, more windows than the context holds: the last entry repeats and the context slides)
  3 effect        another goal gives other codes (right vs left), so the goal really reaches the prior
  4 override      "goal" in cvq_prior_next beats the plan, "goal":null means no goal for that window; a reset without goals clears the old plan
  5 refusals      wrong length / not numbers / null entry in "goals" / a goal for a prior trained without goals are errors, and a refused reset leaves the running session alone
"""
import argparse, ast, importlib.util, json, os, subprocess, sys, tempfile, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
FAILS = []


def check(ok, msg):
    print(("  ok    " if ok else "  FAIL  ") + msg)
    if not ok:
        FAILS.append(msg)


def helpers():
    """The synthetic checkpoint builder of verify_cvqprior.py (that file runs main() on import, so only its functions are taken, with ast) and the real exporter."""
    tree = ast.parse(open(os.path.join(HERE, "verify_cvqprior.py")).read())
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and getattr(n, "name", "") != "main"]
    vc = types.ModuleType("vc"); exec(compile(ast.Module(body=keep, type_ignores=[]), "verify_cvqprior(helpers)", "exec"), vc.__dict__)
    spec = importlib.util.spec_from_file_location("nanity_export_cvqprior", os.path.join(ROOT, "conversion", "nanity_export_cvqprior.py"))
    ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
    return vc, ex


class Neon:
    """neon --motion-model X.gguf --interactive: one JSON line in, lines out until {"done":true}. ask() returns the motion reply or {"error": ...}."""

    def __init__(self, binary, gguf):
        self.p = subprocess.Popen([binary, "--motion-model", gguf, "--interactive"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def ask(self, op, **kw):
        self.p.stdin.write(json.dumps({"motion": dict(op=op, **kw)}) + "\n"); self.p.stdin.flush()
        out = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("neon closed the pipe")
            try:
                j = json.loads(line)
            except ValueError:
                continue
            if "motion" in j:
                out = j["motion"]
            elif "error" in j:
                out = {"error": j["error"]}
            if j.get("done"):
                return out

    def close(self):
        self.p.stdin.close(); self.p.wait(timeout=10)


def run(neon, windows, *, seed, reset=None, per_window=None, nxt=None):
    """reset(extra keys) then `windows` x cvq_prior_next. per_window[i] = the goal key of the i-th next ({} = none given); nxt = keys for every next. -> (reset reply, [codes per window])"""
    r = neon.ask("cvq_prior_reset", seed=seed, action="null", **(reset or {}))
    codes = []
    for i in range(windows):
        kw = dict(temperature=0.9, top_k=0); kw.update(nxt or {}); kw.update((per_window[min(i, len(per_window) - 1)] if per_window else {}))
        j = neon.ask("cvq_prior_next", **kw)
        codes.append(j["codes"] if "codes" in j else j)
    return r, codes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--neon", default=os.environ.get("NEON_BIN", "./neon"), help="a built NEON binary")
    a = ap.parse_args()
    if not os.path.isfile(a.neon):
        sys.exit(f"no NEON binary at {a.neon}: build it (see the top of this file) and pass --neon")
    vc, ex = helpers()
    tmp = tempfile.mkdtemp(prefix="cvqprior_goal_")
    mk = lambda goal: vc.synthetic(argparse.Namespace(K=64, n=16, windows=4, d=64, layers=2, heads=4, actions=7, goal=goal, styles=0))
    gpath, ppath = os.path.join(tmp, "goal.gguf"), os.path.join(tmp, "plain.gguf")
    ex.export_ckpt(mk(True), gpath); ex.export_ckpt(mk(False), ppath)
    neon = Neon(a.neon, gpath)

    L, R, S = [0.5, 0.3, 1.0, 1.1], [-0.5, -0.3, 1.0, 1.1], [0.0, 0.0, 0.0, 0.0]               # [dyaw, dx, dz, speed]: a turn to the left (+, the trainer's convention), to the right (-), and a standing goal
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 1
    print("[info]")
    g = neon.ask("info").get("cvqprior", {}).get("goal")
    check(bool(g) and g["names"] == ["dyaw", "dx", "dz", "speed"] and len(g["mean"]) == 4 and len(g["std"]) == 4, f"info.cvqprior.goal = {g and g['names']}")
    _, base9 = run(neon, 4, seed=9)                                                          # taken on a process that never saw a plan: what "no goal" gives (used in [override / clear])

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 2
    print("[a plan in cvq_prior_reset == the same goals given to every cvq_prior_next]")
    W = 6                                                                                      # more windows than the prior's context (4): the last plan entry repeats and the context slides
    for seed in (5, 11):
        for plan in ([R], [R, L], [R, L, S]):
            r, via_plan = run(neon, W, seed=seed, reset=dict(goals=plan))
            _, via_next = run(neon, W, seed=seed, per_window=[dict(goal=p) for p in plan])
            check(r.get("planned_windows") == len(plan) and via_plan == via_next, f"seed {seed}, {len(plan)} planned goal(s), {W} windows: identical codes (reply: planned_windows = {r.get('planned_windows')})")
    _, one = run(neon, 4, seed=3, reset=dict(goal=L)); _, many = run(neon, 4, seed=3, reset=dict(goals=[L]))
    check(one == many, '"goal" in the reset = "goals" with that one entry (it applies to every window)')
    _, both = run(neon, 4, seed=3, reset=dict(goal=R, goals=[L])); check(both == many, '"goals" wins over "goal" when both are given (as in cvq_sample)')

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 3
    print("[the goal reaches the prior]")
    _, none = run(neon, W, seed=7); _, right = run(neon, W, seed=7, reset=dict(goal=R)); _, left = run(neon, W, seed=7, reset=dict(goal=L))
    check(right != none and left != none and right != left, "same seed: no goal / goal R / goal L give three different code sequences")
    _, right2 = run(neon, W, seed=7, reset=dict(goal=R)); check(right2 == right, "...and the same plan with the same seed repeats exactly")
    _, ramp = run(neon, W, seed=7, reset=dict(goals=[R, R, L, L, L, L]))
    check(ramp[:2] == right[:2] and ramp[2] != right[2], "a plan R,R,L,...: windows 0-1 are as under R alone, window 2 already differs (the goal of window i acts on window i)")

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 4
    print("[override / clear]")
    _, ovr = run(neon, 4, seed=9, reset=dict(goals=[R, R, R, R]), per_window=[dict(), dict(goal=L), dict(goal=None), dict()])
    _, ref = run(neon, 4, seed=9, per_window=[dict(goal=R), dict(goal=L), dict(), dict(goal=R)])
    check(ovr == ref, 'next with "goal" beats the plan (window 1), "goal":null is a window without a goal (window 2), absent falls back to the plan (windows 0 and 3)')
    neon.ask("cvq_prior_reset", seed=9, action="null", goals=[R, L])                          # a plan is set ...
    r, cleared = run(neon, 4, seed=9)                                                        # ... and the next reset has none
    check(r.get("planned_windows") == 0 and cleared == base9, "a reset without goals clears the old plan: planned_windows = 0 and the codes are those of a process that never had one")

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------- 5
    print("[refusals]")
    for bad, why in ((dict(goal=[1, 2]), "a goal of the wrong length"), (dict(goals=[R, None]), "a null entry in goals"), (dict(goals=[R, [1, 2, "x", 4]]), "a non-number in a goal"),
                     (dict(goals=[]), None)):
        r = neon.ask("cvq_prior_reset", seed=1, action="null", **bad)
        if why:
            check("error" in r, f"{why}: refused ({str(r.get('error', ''))[:60]})")
        else:
            check("error" not in r and r.get("planned_windows") == 0, 'an empty "goals" list is no plan')
    _, before = run(neon, 3, seed=21, reset=dict(goals=[R, L]))                              # the uninterrupted run
    neon.ask("cvq_prior_reset", seed=21, action="null", goals=[R, L]); first = [neon.ask("cvq_prior_next", temperature=0.9, top_k=0)["codes"] for _ in range(2)]
    r = neon.ask("cvq_prior_reset", seed=1, action="null", goals=[R, [1, 2]])               # refused in the middle of the session
    nxt = neon.ask("cvq_prior_next", temperature=0.9, top_k=0)
    check("error" in r and nxt.get("window") == 2 and first == before[:2] and nxt.get("codes") == before[2], "a refused reset leaves the running session alone: the next window is still #2, with the same codes and plan")
    neon.close()

    plain_neon = Neon(a.neon, ppath)
    check(plain_neon.ask("info")["cvqprior"].get("goal") is None, "a prior trained without goals reports info.cvqprior.goal = null")
    for kw, nm in ((dict(goal=R), "reset"), (dict(goals=[R]), "reset goals")):
        check("error" in plain_neon.ask("cvq_prior_reset", seed=1, action="null", **kw), f"{nm} with a goal on a prior trained without goals: refused")
    check("error" in plain_neon.ask("cvq_prior_next", goal=R), "next with a goal on a prior trained without goals: refused")
    check("codes" in plain_neon.ask("cvq_prior_next"), "...and it still samples without one")
    plain_neon.close()
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED"); sys.exit(1)
    print("all goal-op checks passed")


main()
