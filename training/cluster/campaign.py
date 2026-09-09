"""The run queue: nine models, one notebook, and a session that will be killed.

A JupyterHub session is not a place to run a training campaign. It is a place
to *supervise* one, badly - the kernel dies on idle timeout, on a wall-clock
limit, on the browser tab closing, and on the cluster reclaiming the node. Any
design where the notebook holds the training state loses that state several
times a week.

So the notebook holds nothing. It calls `step()` in a loop; everything that
matters lives in two places on disk:

* **`campaign_state.json`** - which runs are done, which failed, how many
  attempts each has had, and when each last wrote a heartbeat;
* **the checkpoint directories** - which already survive a kill, because
  `training/common/checkpointing.py` was built for exactly this failure and
  every trainer here uses it.

Restarting the kernel therefore costs the seconds since the last checkpoint,
not the run.

## The stale-run problem, which is the whole reason this file exists

When a session is killed mid-run, nothing gets the chance to write "failed".
The state file says `running` forever, and a naive driver either waits for a
process that is gone or, worse, starts a second copy of a run whose first copy
is still alive on another node - two processes writing the same checkpoint
directory, which corrupts it.

Both halves are handled by a heartbeat plus an owner record. A run marked
`running` is only believed while its heartbeat is fresh AND its owner claims
to be this host and a live pid. Otherwise it is *reclaimed*: marked
interrupted, attempt count incremented, and re-queued with `--resume`. A run
owned by a live process elsewhere is never touched, and the driver says so
rather than skipping silently.

## Why runs are subprocesses and not imports

Three reasons, in order of how much each has cost this project:

1. **A CUDA OOM does not reliably leave an importable process healthy.** In a
   notebook it usually leaves the whole kernel unusable, taking the queue with
   it. A subprocess dies alone and the queue records the exit code.
2. The trainers already have argparse entry points that are unit tested. A
   second, import-shaped entry point would be a second thing to keep correct.
3. Each run gets its own log file, and the log is the artifact you read at
   9 a.m. when you want to know what happened at 3 a.m.

## Budgets are advisory, and that is deliberate

`step(budget_minutes=...)` **prefers** a run that fits the remaining session
and starts a longer one anyway when nothing fits. It never tries to stop a run
already going, because a run killed by the session limit is exactly the case
resume was built for.

The alternative - refusing to start anything longer than the session - sounds
tidier and is unusable here. Sessions are commonly 4-12 hours; the longest run
in this campaign is estimated at 20. A refusing budget would report "nothing
ready to run" indefinitely while nine models stayed untrained. `strict=True`
opts into the refusing behaviour for the case where it is correct: a shared
node that has to be handed back on time.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# A run whose heartbeat is older than this is presumed dead. Trainers touch
# the heartbeat once a minute (see `_heartbeat_path`), so the margin is wide
# enough to survive a slow epoch boundary or a stalled shared filesystem and
# short enough that a session killed at 02:00 is reclaimable by 02:10.
STALE_AFTER_S = 600.0

PENDING, RUNNING, DONE, FAILED, INTERRUPTED = (
    "pending", "running", "done", "failed", "interrupted",
)
# Statuses the queue will pick up. `failed` is deliberately absent: a run that
# raised is a run a human should look at, and silently retrying it burns GPU
# hours reproducing the same traceback. `--retry-failed` opts back in.
RESUMABLE = {PENDING, INTERRUPTED}

# Substrings that make a line exempt from output throttling. Deliberately
# broad: a false positive costs one extra printed line, a false negative
# costs the one traceback that explained why a 14-hour run died.
_URGENT = (
    "error", "traceback", "exception", "failed", "nan", "out of memory",
    "oom", "cuda", "warning: ", "abort", "killed",
)


@dataclass
class RunState:
    id: str
    status: str = PENDING
    attempts: int = 0
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    heartbeat: float | None = None
    owner_host: str | None = None
    owner_pid: int | None = None
    seconds: float = 0.0
    log: str | None = None
    error: str | None = None

    def is_stale(self, now: float | None = None) -> bool:
        """Marked running, but nothing is running it."""
        if self.status != RUNNING:
            return False
        now = now if now is not None else time.time()
        beat = self.heartbeat or self.started_at or 0.0
        if now - beat < STALE_AFTER_S:
            return False
        return not self.owner_alive()

    def owner_alive(self) -> bool:
        """Is the recorded owner process still running, on this host?

        A different host cannot be checked from here, so a foreign owner with
        a fresh heartbeat is reported alive and left alone. That is the safe
        direction: refusing to start a duplicate costs a wait, starting one
        costs a checkpoint directory.
        """
        if self.owner_pid is None:
            return False
        if self.owner_host and self.owner_host != _hostname():
            return (time.time() - (self.heartbeat or 0.0)) < STALE_AFTER_S
        try:
            os.kill(self.owner_pid, 0)
        except (OSError, ProcessLookupError):
            return False
        except PermissionError:
            # The pid exists and belongs to somebody else. Alive, not ours.
            return True
        return True


@dataclass
class Run:
    """One entry from `configs/campaign.yaml`."""

    id: str
    tool: str
    family: str
    script: str
    args: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    needs_data: list[str] = field(default_factory=list)
    ckpt_dir: str = ""
    est_hours: float = 1.0
    resume_flag: str = "--resume"
    note: str = ""

    def command(self, python: str, resume: bool, extra: list[str] | None = None) -> list[str]:
        cmd = [python, str(REPO_ROOT / self.script), *self.args, *(extra or [])]
        if resume and self.resume_flag:
            cmd.append(self.resume_flag)
        return cmd


def _hostname() -> str:
    return os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME") or "unknown"


def _heartbeat_path(state_dir: Path, run_id: str) -> Path:
    """Where a running trainer touches its liveness marker.

    A separate file per run rather than a field in the shared state file: two
    processes rewriting one JSON document is a lost-update race, and the state
    file is the thing that must never be corrupted.
    """
    return state_dir / "heartbeats" / f"{run_id}.beat"


class Campaign:
    """Load the config, hold the state, hand out the next thing to run."""

    def __init__(self, config: dict, state_dir: Path):
        self.version = config.get("version", "unversioned")
        self.name = config.get("name", "campaign")
        self.checkpoint_root = config.get("checkpoint_root", "checkpoints/v2")
        self.runs: dict[str, Run] = {}
        for entry in config["runs"]:
            run = Run(**entry)
            if not run.ckpt_dir:
                run.ckpt_dir = f"{self.checkpoint_root}/{run.id}"
            self.runs[run.id] = run
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "campaign_state.json"
        self.state: dict[str, RunState] = {}
        self._validate()
        self.load()

    # --- config validation --------------------------------------------------

    def _validate(self) -> None:
        """Fail on a broken config now, not eight hours into the third run."""
        for run in self.runs.values():
            for dep in run.depends_on:
                if dep not in self.runs:
                    raise ValueError(f"run '{run.id}' depends on unknown run '{dep}'")
            script = REPO_ROOT / run.script
            if not script.is_file():
                raise FileNotFoundError(
                    f"run '{run.id}' names {run.script}, which does not exist"
                )
        self._toposort()  # raises on a cycle

    def _toposort(self) -> list[str]:
        ordered: list[str] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(node: str, trail: tuple[str, ...]) -> None:
            if node in permanent:
                return
            if node in temporary:
                cycle = " -> ".join(trail + (node,))
                raise ValueError(f"dependency cycle in campaign config: {cycle}")
            temporary.add(node)
            for dep in self.runs[node].depends_on:
                visit(dep, trail + (node,))
            temporary.discard(node)
            permanent.add(node)
            ordered.append(node)

        for node in self.runs:
            visit(node, ())
        return ordered

    # --- state --------------------------------------------------------------

    def load(self) -> None:
        if self.state_path.is_file():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state = {k: RunState(**v) for k, v in raw.get("runs", {}).items()}
        for run_id in self.runs:
            self.state.setdefault(run_id, RunState(id=run_id))

    def save(self) -> None:
        """Atomic write. A driver killed mid-save must not eat the state file."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "campaign": self.name,
            "version": self.version,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "runs": {k: asdict(v) for k, v in self.state.items()},
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def reclaim_stale(self, verbose: bool = True) -> list[str]:
        """Re-queue runs whose session died without writing an outcome."""
        reclaimed = []
        for run_id, state in self.state.items():
            beat = _heartbeat_path(self.state_dir, run_id)
            if state.status == RUNNING and beat.is_file():
                # Trust the file over the field: the field was written when
                # the run started, the file is written while it lives.
                state.heartbeat = max(state.heartbeat or 0.0, beat.stat().st_mtime)
            if state.is_stale():
                state.status = INTERRUPTED
                state.error = "session ended without an outcome; resuming"
                state.owner_pid = None
                reclaimed.append(run_id)
                if verbose:
                    print(f"reclaimed '{run_id}' (stale heartbeat) - will resume")
        if reclaimed:
            self.save()
        return reclaimed

    # --- scheduling ---------------------------------------------------------

    def ready(self) -> list[str]:
        """Runs that could start now, in dependency order."""
        out = []
        for run_id in self._toposort():
            state = self.state[run_id]
            if state.status not in RESUMABLE:
                continue
            deps = self.runs[run_id].depends_on
            if all(self.state[d].status == DONE for d in deps):
                out.append(run_id)
        return out

    def remaining_minutes(self, run_id: str) -> float:
        """How much longer this run is expected to need.

        Crude - elapsed against the estimate - but strictly better than
        treating a run that is 90% done as a fresh eight-hour commitment and
        never starting it in a short session.
        """
        run, state = self.runs[run_id], self.state[run_id]
        return max(0.0, run.est_hours * 60 - state.seconds / 60)

    def next_run(self, budget_minutes: float | None = None,
                 strict: bool = False) -> str | None:
        """The next run to start, or None.

        A budget PREFERS a run that fits; it does not refuse to start one that
        does not. That distinction is the whole design:

        every trainer here checkpoints every few hundred steps, so a run
        killed by the session limit loses seconds, not hours. Meanwhile a
        JupyterHub session is commonly 4-12 hours and the longest run in this
        campaign is estimated at 20. A budget that blocked anything longer
        than the session would train **nothing, ever** - the campaign would
        report "nothing ready to run" forever while nine models stayed
        untrained. That is not a hypothetical; it is what the first version of
        this method did.

        So the ordering is: fitting runs first, longest-progress run
        otherwise. `strict=True` restores the refusing behaviour for the one
        case where it is right - a shared node you must hand back on time.
        """
        candidates = self.ready()
        if not candidates:
            return None
        if budget_minutes is None:
            return candidates[0]

        fitting = [r for r in candidates
                   if self.remaining_minutes(r) <= budget_minutes]
        if fitting:
            return fitting[0]
        if strict:
            return None
        # Nothing fits. Start the one closest to finishing anyway - it makes
        # the most progress per session and is the most likely to complete in
        # the next one.
        return min(candidates, key=self.remaining_minutes)

    # --- execution ----------------------------------------------------------

    def launch(self, run_id: str, python: str = sys.executable,
               extra: list[str] | None = None, dry_run: bool = False,
               echo: str = "throttled", echo_every_s: float = 30.0) -> int:
        """Run one job to completion in a subprocess, streaming its log.

        `echo` controls what reaches the caller's stdout. The log file always
        gets every line regardless; this is only about what is *displayed*.

        * `"throttled"` (default) - at most one progress line per
          `echo_every_s`, plus every line that looks like a failure. This is
          the default because of what the alternative does in a notebook: a
          14-hour run emits tens of thousands of lines, every one of which
          Jupyter stores in the `.ipynb` as cell output. The file grows to
          hundreds of megabytes, the browser tab becomes unusable, and the
          run you were watching is now the run you cannot see.
        * `"full"` - every line. Correct in a terminal, wrong in a notebook.
        * `"none"` - nothing but the start and end banners.
        """
        run, state = self.runs[run_id], self.state[run_id]

        if state.status == RUNNING and state.owner_alive():
            print(f"'{run_id}' is already running (host={state.owner_host} "
                  f"pid={state.owner_pid}); refusing to start a second copy")
            return 1

        resume = state.attempts > 0
        cmd = run.command(python, resume=resume, extra=extra)

        log_dir = self.state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"

        if dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            return 0

        state.status = RUNNING
        state.attempts += 1
        state.started_at = time.time()
        state.heartbeat = state.started_at
        state.owner_host = _hostname()
        state.owner_pid = os.getpid()
        state.log = str(log_path)
        state.error = None
        self.save()

        beat = _heartbeat_path(self.state_dir, run_id)
        beat.parent.mkdir(parents=True, exist_ok=True)
        beat.touch()

        print(f"=== {run_id} (attempt {state.attempts}{', resuming' if resume else ''}) ===")
        print(f"    {' '.join(cmd)}")
        print(f"    log: {log_path}")

        started = time.time()
        last_beat = started
        code = 1
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write(f"\n\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                          f"attempt {state.attempts} ===\n{' '.join(cmd)}\n\n")
                log.flush()
                proc = subprocess.Popen(
                    cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                    errors="replace",
                )
                last_echo = 0.0
                suppressed = 0
                for line in proc.stdout:
                    log.write(line)
                    now = time.time()

                    if echo == "full":
                        print(line, end="")
                    elif echo == "throttled":
                        # A line naming a failure is never throttled away.
                        # Losing the one traceback in 40,000 lines of progress
                        # is the whole cost of getting this wrong.
                        urgent = any(word in line.lower() for word in _URGENT)
                        if urgent or now - last_echo >= echo_every_s:
                            prefix = f"    [+{suppressed}] " if suppressed else "    "
                            print(prefix + line.rstrip())
                            last_echo, suppressed = now, 0
                        else:
                            suppressed += 1

                    if now - last_beat > 60:
                        # Touched from the driver rather than from inside the
                        # trainer: the trainer is third-party-shaped code that
                        # should not have to know about the queue, and a
                        # trainer that has stopped producing output is a
                        # trainer that is stuck, which is what we want to see.
                        beat.touch()
                        state.heartbeat = now
                        last_beat = now
                        log.flush()
                code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            state.status = INTERRUPTED
            state.error = "interrupted by the operator"
            state.seconds += time.time() - started
            state.owner_pid = None
            self.save()
            print(f"\n'{run_id}' interrupted; rerun to resume")
            raise
        finally:
            state.seconds += time.time() - started
            state.finished_at = time.time()
            state.owner_pid = None

        state.exit_code = code
        state.status = DONE if code == 0 else FAILED
        if code != 0:
            state.error = f"exit code {code}; see {log_path}"
        self.save()

        elapsed = (state.finished_at - started) / 60
        print(f"=== {run_id} {state.status} in {elapsed:.1f} min (exit {code}) ===")
        return code

    def step(self, budget_minutes: float | None = None,
             python: str = sys.executable, dry_run: bool = False,
             strict: bool = False, echo: str = "throttled") -> str | None:
        """Reclaim, pick one run, execute it. The notebook's entire API."""
        self.reclaim_stale()
        run_id = self.next_run(budget_minutes, strict=strict)
        if run_id is not None and budget_minutes is not None:
            remaining = self.remaining_minutes(run_id)
            if remaining > budget_minutes:
                print(f"'{run_id}' needs ~{remaining:.0f} min and the session "
                      f"budget is {budget_minutes:.0f}; starting it anyway - it "
                      "checkpoints and the next session resumes it.")
        if run_id is None:
            return None
        self.launch(run_id, python=python, dry_run=dry_run, echo=echo)
        return run_id

    def run_all(self, budget_minutes: float | None = None,
                python: str = sys.executable, dry_run: bool = False,
                strict: bool = False, echo: str = "throttled") -> int:
        """Keep stepping until nothing is ready. Returns the number of failures."""
        started = time.time()
        while True:
            remaining = None
            if budget_minutes is not None:
                remaining = budget_minutes - (time.time() - started) / 60
                if remaining <= 0:
                    print("session budget spent; stopping before the next run")
                    break
            if self.step(remaining, python=python, dry_run=dry_run,
                         strict=strict, echo=echo) is None:
                break
        return sum(1 for s in self.state.values() if s.status == FAILED)

    # --- monitoring a run this process is not running -----------------------

    def log_path(self, run_id: str) -> Path:
        return self.state_dir / "logs" / f"{run_id}.log"

    def tail(self, run_id: str, lines: int = 40) -> str:
        """The last `lines` of a run's log.

        This is what makes the detached mode usable. When the campaign is
        launched with `nohup` from a terminal, the notebook is not its parent
        and never sees its stdout - but the log file is on a shared
        filesystem, so a monitoring cell can read it. Re-running that cell is
        the notebook equivalent of `tail -f`, without holding a cell open for
        fourteen hours.
        """
        path = self.log_path(run_id)
        if not path.is_file():
            return f"(no log yet at {path})"
        # Read the tail rather than the file: a long run's log reaches
        # hundreds of MB, and reading all of it to show 40 lines is how a
        # monitoring cell becomes the thing that kills the kernel.
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            block = min(size, max(4096, lines * 400))
            fh.seek(size - block)
            text = fh.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def watch(self, lines: int = 20) -> str:
        """One screen showing every active run and the tail of its log."""
        self.reclaim_stale(verbose=False)
        out = [self.report()]
        for run_id, state in self.state.items():
            if state.status == RUNNING:
                age = time.time() - (state.heartbeat or state.started_at or 0)
                out += ["", f"--- {run_id} (heartbeat {age:.0f}s ago) ---",
                        self.tail(run_id, lines)]
        return "\n".join(out)

    # --- reporting ----------------------------------------------------------

    def report(self) -> str:
        lines = [f"{self.name}  ({self.version})", ""]
        lines.append(f"{'run':<18} {'tool':<20} {'status':<12} {'attempts':<9} "
                     f"{'hours':<7} depends on")
        done = 0
        for run_id in self._toposort():
            run, state = self.runs[run_id], self.state[run_id]
            if state.status == DONE:
                done += 1
            hours = state.seconds / 3600
            lines.append(
                f"{run_id:<18} {run.tool:<20} {state.status:<12} "
                f"{state.attempts:<9} {hours:<7.2f} {', '.join(run.depends_on) or '-'}"
            )
            if state.error:
                lines.append(f"    {state.error}")
        total = sum(s.seconds for s in self.state.values()) / 3600
        estimated = sum(r.est_hours for r in self.runs.values())
        lines += [
            "",
            f"{done}/{len(self.runs)} complete   "
            f"{total:.1f} GPU-h spent of ~{estimated:.0f} estimated",
        ]
        return "\n".join(lines)


def load_campaign(config_path: str | Path = "configs/campaign.yaml",
                  state_dir: str | Path = "runs") -> Campaign:
    import yaml

    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    state = Path(state_dir)
    if not state.is_absolute():
        state = REPO_ROOT / state
    return Campaign(config, state)


def main() -> int:
    p = argparse.ArgumentParser(description="Drive the Phase 5 training campaign.")
    p.add_argument("--config", default="configs/campaign.yaml")
    p.add_argument("--state-dir", default="runs")
    p.add_argument("--budget-minutes", type=float,
                   help="do not start a run expected to outlast this")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--echo", choices=["full", "throttled", "none"],
                   default="full",
                   help="how much of a run's output to display. Defaults to "
                        "full on the command line, where scrollback is cheap; "
                        "the notebook passes throttled, where it is not")
    p.add_argument("--strict-budget", action="store_true",
                   help="refuse to start a run longer than the budget, rather "
                        "than starting it and letting the next session resume it")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="what is done, running and pending")
    w = sub.add_parser("watch", help="status plus the tail of each running log")
    w.add_argument("--lines", type=int, default=20)
    tl = sub.add_parser("tail", help="the tail of one run's log")
    tl.add_argument("run_id")
    tl.add_argument("--lines", type=int, default=40)
    sub.add_parser("step", help="run the next ready job")
    sub.add_parser("all", help="run every ready job until none remain")
    one = sub.add_parser("run", help="run one job by id")
    one.add_argument("run_id")
    reset = sub.add_parser("reset", help="return a run to pending")
    reset.add_argument("run_id")
    reset.add_argument("--keep-attempts", action="store_true",
                       help="keep the attempt count, so the rerun still resumes")

    args = p.parse_args()
    campaign = load_campaign(args.config, args.state_dir)

    if args.command == "status":
        campaign.reclaim_stale()
        print(campaign.report())
        return 0

    if args.command == "watch":
        print(campaign.watch(args.lines))
        return 0

    if args.command == "tail":
        print(campaign.tail(args.run_id, args.lines))
        return 0

    if args.command == "reset":
        state = campaign.state[args.run_id]
        state.status = PENDING
        state.error = None
        state.exit_code = None
        if not args.keep_attempts:
            state.attempts = 0
            state.seconds = 0.0
        campaign.save()
        print(f"'{args.run_id}' reset to pending "
              f"({'will resume' if args.keep_attempts else 'will start fresh'})")
        return 0

    if args.command == "run":
        campaign.reclaim_stale()
        return campaign.launch(args.run_id, python=args.python,
                               dry_run=args.dry_run, echo=args.echo)

    if args.command == "step":
        run_id = campaign.step(args.budget_minutes, python=args.python,
                               dry_run=args.dry_run, strict=args.strict_budget,
                               echo=args.echo)
        if run_id is None:
            print("nothing ready to run")
            print(campaign.report())
        return 0

    failures = campaign.run_all(args.budget_minutes, python=args.python,
                                dry_run=args.dry_run, strict=args.strict_budget,
                                echo=args.echo)
    print()
    print(campaign.report())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
