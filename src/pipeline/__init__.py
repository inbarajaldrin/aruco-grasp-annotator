"""
Deterministic CAD -> outputs pipeline orchestrator.

Drop in an assembly config + its meshes, get wireframe + markers (ArUco or AprilTag, with
PNG/PDF cut-sheets) + fold symmetry + grasp points, in one self-describing job folder. All
stages are in-process core calls; the orchestrator is sequencing + provenance.

Entry points:
    run_job(spec: JobSpec) -> JobResult     # programmatic
    aruco-pipeline                          # console script (pipeline.cli:main)
"""

from .orchestrator import run_job
from .spec import AssemblySpec, JobResult, JobSpec

__all__ = ["run_job", "JobSpec", "AssemblySpec", "JobResult"]
