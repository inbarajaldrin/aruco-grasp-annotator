"""
aruco-pipeline: CLI front-end for run_job.

Two ways to specify a job:
  * --spec job.json                          (a serialized JobSpec; flags below override it)
  * inline flags                             (--output-dir/--models-dir/--assembly/...)

Examples:
  # all 3 fmb assemblies, ArUco, into a job folder
  aruco-pipeline --output-dir data_v2/jobs/fmb --models-dir data/models \\
      --assembly-glob 'data/fmb_assembly*.json'

  # one assembly, AprilTag, markers + grasp only
  aruco-pipeline --output-dir /tmp/job --models-dir data/models \\
      --assembly fmb1=data/fmb_assembly1.json --marker-family apriltag --stages markers,grasp

  # prove determinism against an existing job folder
  aruco-pipeline --spec job.json --mode verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .orchestrator import run_job
from .spec import ALL_STAGES, AssemblySpec, JobResult, JobSpec


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aruco-pipeline", description="Deterministic CAD->outputs pipeline.")
    p.add_argument("--spec", type=Path, help="Load a JobSpec from JSON (flags override).")
    p.add_argument("--output-dir", type=Path, help="Job output folder.")
    p.add_argument("--models-dir", type=Path, help="Directory of object meshes (.obj/.stl/.ply).")
    p.add_argument("--assembly", action="append", default=[], metavar="NAME=CONFIG",
                   help="An assembly as name=path/to/config.json (repeatable).")
    p.add_argument("--assembly-glob", help="Glob of assembly configs; each named by file stem.")
    p.add_argument("--marker-family", choices=["aruco", "apriltag"], help="Fiducial family (default aruco).")
    p.add_argument("--border-width", type=float, help="Quiet-zone fraction override (default: family default).")
    p.add_argument("--stages", help=f"Comma list to run (default all: {','.join(ALL_STAGES)}).")
    p.add_argument("--skip", help="Comma list of stages to skip.")
    p.add_argument("--mode", choices=["rebuild", "skip_existing", "verify"], help="Run mode (default rebuild).")
    p.add_argument("--bundle-models", action="store_true", help="Copy meshes into the job folder (portable archive).")
    return p


def _assemblies_from_args(args: argparse.Namespace) -> list[AssemblySpec]:
    out: list[AssemblySpec] = []
    for item in args.assembly:
        if "=" not in item:
            raise SystemExit(f"--assembly must be NAME=CONFIG, got {item!r}")
        name, _, cfg = item.partition("=")
        out.append(AssemblySpec(name=name, config=Path(cfg)))
    if args.assembly_glob:
        for cfg in sorted(Path().glob(args.assembly_glob)):
            out.append(AssemblySpec(name=cfg.stem, config=cfg))
    return out


def build_spec(args: argparse.Namespace) -> JobSpec:
    if args.spec:
        spec = JobSpec.from_json_file(args.spec)
        updates: dict = {}
        if args.output_dir:
            updates["output_dir"] = args.output_dir
        if args.models_dir:
            updates["models_dir"] = args.models_dir
        if args.marker_family:
            updates["marker_family"] = args.marker_family
        if args.border_width is not None:
            updates["border_width"] = args.border_width
        if args.stages:
            updates["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]
        if args.skip:
            updates["skip"] = [s.strip() for s in args.skip.split(",") if s.strip()]
        if args.mode:
            updates["mode"] = args.mode
        if args.bundle_models:
            updates["bundle_models"] = True
        extra = _assemblies_from_args(args)
        if extra:
            updates["assemblies"] = [a.model_dump() for a in extra]
        if not updates:
            return spec
        # Re-validate the merged spec so bad overrides (e.g. --stages markerz) are rejected
        # loudly rather than silently ignored by active_stages().
        return JobSpec.model_validate({**spec.model_dump(), **updates})

    if not (args.output_dir and args.models_dir):
        raise SystemExit("without --spec you must pass --output-dir and --models-dir")
    assemblies = _assemblies_from_args(args)
    if not assemblies:
        raise SystemExit("no assemblies: pass --assembly NAME=CONFIG or --assembly-glob")
    kwargs: dict = dict(output_dir=args.output_dir, models_dir=args.models_dir, assemblies=assemblies)
    if args.marker_family:
        kwargs["marker_family"] = args.marker_family
    if args.border_width is not None:
        kwargs["border_width"] = args.border_width
    if args.stages:
        kwargs["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.skip:
        kwargs["skip"] = [s.strip() for s in args.skip.split(",") if s.strip()]
    if args.mode:
        kwargs["mode"] = args.mode
    if args.bundle_models:
        kwargs["bundle_models"] = True
    return JobSpec(**kwargs)


def _report(spec: JobSpec, result: JobResult) -> None:
    print(f"\nJob: {spec.output_dir}  (family={spec.marker_family}, mode={spec.mode})")
    for a in result.assemblies:
        counts: dict[str, list[int]] = {}  # stage -> [ok, skip, fail]
        for s in a.stages:
            c = counts.setdefault(s.stage, [0, 0, 0])
            c[{"ok": 0, "skip": 1, "fail": 2}[s.status]] += 1
        dict_note = f" dict={a.marker_dictionary} markers={a.total_markers}" if a.marker_dictionary else ""
        print(f"  [{a.name}]{dict_note}")
        for stage in ALL_STAGES:
            if stage in counts:
                ok, sk, fa = counts[stage]
                print(f"      {stage:13} ok={ok} skip={sk} fail={fa}")
        for w in a.warnings:
            print(f"      ! {w}")
        for s in a.stages:
            if s.status == "fail":
                print(f"      FAIL {s.stage}/{s.object}: {s.detail}")
    print(f"\n{'OK' if result.ok else 'FAILED'} — {result.failed} stage failure(s)")


def main() -> int:
    from pydantic import ValidationError

    args = _build_parser().parse_args()
    try:
        spec = build_spec(args)
    except ValidationError as exc:
        print(f"invalid job spec:\n{exc}", file=sys.stderr)
        return 2
    result = run_job(spec)
    _report(spec, result)
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
