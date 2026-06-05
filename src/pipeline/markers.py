"""
Marker-family registry: the one place that knows how ArUco and AprilTag differ.

The placement backbone (which face, seating, size, pose) is fiducial-agnostic — it does
not care whether the printed tag is ArUco or AprilTag. Only three things are family-
specific, and they all live here:

  1. the candidate dictionaries + their ID capacities (the auto-select ladder),
  2. the default border / quiet-zone ratio used when rendering, and
  3. which dictionary the assembly's total marker count is allowed to use.

OpenCV's ``cv2.aruco`` ships AprilTag families as predefined dictionaries
(``DICT_APRILTAG_36h11`` etc.), so the *renderer* is shared too — see
``aruco_annotator.scripts.generate_aruco_png`` / ``...aruco_utils``, whose dictionary
maps are extended to include the AprilTag entries this module references.

This module is deliberately dependency-free (no cv2 / open3d): the contiguous-ID and
dictionary-selection logic is pure arithmetic and unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarkerFamily:
    """A fiducial family: its name, its dictionary ladder, and its rendering defaults.

    ``ladder`` is ordered ascending by capacity. The first entry is the *robust default*
    (largest inter-marker Hamming distance for the family); selecting anything past it
    trades robustness for capacity and emits a warning.
    """

    name: str
    ladder: tuple[tuple[str, int], ...]  # (dict_name, id_capacity), ascending capacity
    default_border: float                # white quiet-zone fraction inside `size`

    def dictionaries(self) -> list[str]:
        return [d for d, _ in self.ladder]


# ArUco: stay in the 4X4 family for detection robustness; ladder up only when an assembly
# needs more than 50 markers. (5X5/6X6/7X7 exist in the renderer but are not used for
# auto-selection — denser bit grids print larger for the same robustness.)
_ARUCO = MarkerFamily(
    name="aruco",
    ladder=(
        ("DICT_4X4_50", 50),
        ("DICT_4X4_100", 100),
        ("DICT_4X4_250", 250),
        ("DICT_4X4_1000", 1000),
    ),
    default_border=0.05,
)

# AprilTag: 36h11 is the robotics standard — 587 IDs (covers any realistic assembly) with
# a robust 36-bit code. The smaller families (16h5=30, 25h9=35) are intentionally NOT in
# the ladder: their tiny Hamming distance makes them fragile and they buy little capacity.
#
# NOTE on border: AprilTags carry a proportionally wider black border than ArUco, so the
# `size -> detectable-region` relation the downstream localizer uses differs. The default
# below mirrors ArUco's quiet-zone convention; override per-job via JobSpec.border_width
# and tune the localizer's TOTAL_MARKER_SIZE accordingly before field use.
_APRILTAG = MarkerFamily(
    name="apriltag",
    ladder=(("DICT_APRILTAG_36h11", 587),),
    default_border=0.05,
)

FAMILIES: dict[str, MarkerFamily] = {f.name: f for f in (_ARUCO, _APRILTAG)}


def get_family(name: str) -> MarkerFamily:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"unknown marker family {name!r}; choose one of {sorted(FAMILIES)}"
        ) from None


def select_dictionary(family_name: str, total_markers: int) -> tuple[str, list[str]]:
    """Pick the smallest dictionary in ``family`` that holds ``total_markers`` IDs.

    Returns ``(dict_name, warnings)``. Raises ``ValueError`` if the assembly needs more
    markers than the family's largest dictionary can represent (split the assembly).
    """
    family = get_family(family_name)
    warnings: list[str] = []
    if total_markers <= 0:
        # No markers (e.g. an all-symmetry-only run); use the robust default, no warning.
        return family.ladder[0][0], warnings

    for index, (dict_name, capacity) in enumerate(family.ladder):
        if total_markers <= capacity:
            if index > 0:
                robust = family.ladder[0][0]
                warnings.append(
                    f"assembly needs {total_markers} markers (> {robust} capacity); "
                    f"using {dict_name} — traded detection robustness for capacity"
                )
            return dict_name, warnings

    largest, capacity = family.ladder[-1]
    raise ValueError(
        f"assembly needs {total_markers} markers but the largest {family.name} "
        f"dictionary {largest} holds only {capacity}; split the assembly"
    )
