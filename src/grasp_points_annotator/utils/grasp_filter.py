"""Grasp Point Filter - Geometric Pre-filtering for Robotic Grasping

This module filters grasp points based on gripper width constraints.
It determines which grasp points are geometrically feasible for a parallel-jaw gripper.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


class GraspFilter:
    """Filters grasp points based on gripper constraints"""

    def __init__(
        self,
        gripper_max_width_mm: float = 70.0,
        gripper_half_open_width_mm: float = 30.0,
        safety_margin_mm: float = 0.0,
        gripper_tip_thickness_mm: float = 20.0,
        max_gap_px: int = 20,
        image_size: int = 1024,
        scale_factor: float = 1.2,
        symmetry_tolerance_mm: float = 10.0
    ):
        """
        Initialize the grasp filter.

        Args:
            gripper_max_width_mm: Maximum gripper opening (actual clearance during closing)
            gripper_half_open_width_mm: Half-open gripper width
            safety_margin_mm: Safety margin for open gripper
            gripper_tip_thickness_mm: Thickness of gripper finger tips
            max_gap_px: Adjacency threshold - regions within this gap are considered connected
            image_size: Rendered image size in pixels
            scale_factor: Scale factor used during rendering
            symmetry_tolerance_mm: Tolerance for symmetric grasp check
        """
        self.gripper_max_width_mm = gripper_max_width_mm
        self.gripper_half_open_width_mm = gripper_half_open_width_mm
        self.safety_margin_mm = safety_margin_mm
        self.gripper_tip_thickness_mm = gripper_tip_thickness_mm
        self.max_gap_px = max_gap_px
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.symmetry_tolerance_mm = symmetry_tolerance_mm

        # Calculate half widths (finger extension from center)
        self.half_open_mm = (gripper_max_width_mm / 2) - safety_margin_mm
        self.half_half_open_mm = gripper_half_open_width_mm / 2

    def compute_pixel_to_mm(self, grasp_data: Dict[str, Any]) -> float:
        """Compute pixel to millimeter conversion factor from wireframe data."""
        vertices = np.array(grasp_data['wireframe']['vertices'])
        x_range = vertices[:, 0].max() - vertices[:, 0].min()
        y_range = vertices[:, 1].max() - vertices[:, 1].min()
        max_dim_m = max(x_range, y_range) * self.scale_factor
        return (max_dim_m / self.image_size) * 1000

    def check_x_grasp(
        self,
        region: Dict[str, Any],
        all_regions: List[Dict[str, Any]],
        pixel_to_mm: float,
        gripper_half_mm: float
    ) -> Tuple[bool, float, List[int]]:
        """
        Check if X-axis grasp is possible with given gripper half-width.

        Returns:
            Tuple of (valid, required_clearance_mm, blocking_regions)
        """
        grasp_center_x = region['center_x']
        grasp_center_y = region['center_y']
        r_top, r_bottom = region['top'], region['bottom']
        r_left, r_right = region['left'], region['right']

        gripper_half_px = gripper_half_mm / pixel_to_mm

        # Gripper finger range
        finger_left = grasp_center_x - gripper_half_px
        finger_right = grasp_center_x + gripper_half_px

        # Start with the grasp region's own clearance requirement
        left_clearance_px = grasp_center_x - region['left']
        right_clearance_px = region['right'] - grasp_center_x

        blocking_regions = [region['id']]
        blocking_region_objs = [region]

        for other in all_regions:
            if other['id'] == region['id']:
                continue

            o_top, o_bottom = other['top'], other['bottom']
            o_left, o_right = other['left'], other['right']

            # Check Y overlap (same horizontal band)
            y_overlap = not (o_bottom < r_top or o_top > r_bottom)
            if not y_overlap:
                continue

            # Check if this region blocks the grasp
            x_intersect = (finger_left < o_right) and (finger_right > o_left)
            adjacent_left = (o_right < r_left) and (r_left - o_right <= self.max_gap_px)
            adjacent_right = (o_left > r_right) and (o_left - r_right <= self.max_gap_px)

            if x_intersect or adjacent_left or adjacent_right:
                blocking_regions.append(other['id'])
                blocking_region_objs.append(other)
                # Update clearance to include this region's extent
                if o_left < grasp_center_x:
                    left_clearance_px = max(left_clearance_px, grasp_center_x - o_left)
                if o_right > grasp_center_x:
                    right_clearance_px = max(right_clearance_px, o_right - grasp_center_x)

        max_clearance_px = max(left_clearance_px, right_clearance_px)
        max_clearance_mm = max_clearance_px * pixel_to_mm
        left_clearance_mm = left_clearance_px * pixel_to_mm
        right_clearance_mm = right_clearance_px * pixel_to_mm

        # Calculate available height for tip support
        tip_regions = self._find_tip_regions_x(
            region, blocking_region_objs, all_regions, gripper_half_px, grasp_center_x, grasp_center_y
        )

        # Get the Y-extent of all tip regions combined
        blocking_top = min(tip_regions, key=lambda r: r['top'])['top']
        blocking_bottom = max(tip_regions, key=lambda r: r['bottom'])['bottom']

        # Calculate extent from grasp center
        extent_above = grasp_center_y - blocking_top
        extent_below = blocking_bottom - grasp_center_y

        # Calculate available height based on symmetry/edge cases
        if extent_above == 0:
            available_height_mm = extent_below * pixel_to_mm
        elif extent_below == 0:
            available_height_mm = extent_above * pixel_to_mm
        elif extent_above * pixel_to_mm >= self.gripper_tip_thickness_mm / 2 or \
             extent_below * pixel_to_mm >= self.gripper_tip_thickness_mm / 2:
            available_height_mm = (extent_above + extent_below) * pixel_to_mm
        else:
            symmetric_extent_px = min(extent_above, extent_below)
            available_height_mm = symmetric_extent_px * 2 * pixel_to_mm

        # Check symmetry
        is_symmetric = abs(left_clearance_mm - right_clearance_mm) <= self.symmetry_tolerance_mm

        valid = (max_clearance_mm <= gripper_half_mm and
                 is_symmetric and
                 available_height_mm >= self.gripper_tip_thickness_mm)

        return valid, max_clearance_mm, blocking_regions

    def check_y_grasp(
        self,
        region: Dict[str, Any],
        all_regions: List[Dict[str, Any]],
        pixel_to_mm: float,
        gripper_half_mm: float
    ) -> Tuple[bool, float, List[int]]:
        """
        Check if Y-axis grasp is possible with given gripper half-width.

        Returns:
            Tuple of (valid, required_clearance_mm, blocking_regions)
        """
        grasp_center_y = region['center_y']
        grasp_center_x = region['center_x']
        r_left, r_right = region['left'], region['right']
        r_top, r_bottom = region['top'], region['bottom']

        gripper_half_px = gripper_half_mm / pixel_to_mm

        # Gripper finger range
        finger_top = grasp_center_y - gripper_half_px
        finger_bottom = grasp_center_y + gripper_half_px

        # Start with the grasp region's own clearance requirement
        top_clearance_px = grasp_center_y - region['top']
        bottom_clearance_px = region['bottom'] - grasp_center_y

        blocking_regions = [region['id']]
        blocking_region_objs = [region]

        for other in all_regions:
            if other['id'] == region['id']:
                continue

            o_left, o_right = other['left'], other['right']
            o_top, o_bottom = other['top'], other['bottom']

            # Check X overlap (same vertical band)
            x_overlap = not (o_right < r_left or o_left > r_right)
            if not x_overlap:
                continue

            # Check if this region blocks the grasp
            y_intersect = (finger_top < o_bottom) and (finger_bottom > o_top)
            adjacent_top = (o_bottom < r_top) and (r_top - o_bottom <= self.max_gap_px)
            adjacent_bottom = (o_top > r_bottom) and (o_top - r_bottom <= self.max_gap_px)

            if y_intersect or adjacent_top or adjacent_bottom:
                blocking_regions.append(other['id'])
                blocking_region_objs.append(other)
                # Update clearance to include this region's extent
                if o_top < grasp_center_y:
                    top_clearance_px = max(top_clearance_px, grasp_center_y - o_top)
                if o_bottom > grasp_center_y:
                    bottom_clearance_px = max(bottom_clearance_px, o_bottom - grasp_center_y)

        max_clearance_px = max(top_clearance_px, bottom_clearance_px)
        max_clearance_mm = max_clearance_px * pixel_to_mm
        top_clearance_mm = top_clearance_px * pixel_to_mm
        bottom_clearance_mm = bottom_clearance_px * pixel_to_mm

        # Calculate available width for tip support
        tip_regions = self._find_tip_regions_y(
            region, blocking_region_objs, all_regions, gripper_half_px, grasp_center_x, grasp_center_y
        )

        # Get the X-extent of all tip regions combined
        blocking_left = min(tip_regions, key=lambda r: r['left'])['left']
        blocking_right = max(tip_regions, key=lambda r: r['right'])['right']

        # Calculate extent from grasp center
        extent_left = grasp_center_x - blocking_left
        extent_right = blocking_right - grasp_center_x

        # Calculate available width based on symmetry/edge cases
        if extent_left == 0:
            available_width_mm = extent_right * pixel_to_mm
        elif extent_right == 0:
            available_width_mm = extent_left * pixel_to_mm
        elif extent_left * pixel_to_mm >= self.gripper_tip_thickness_mm / 2 or \
             extent_right * pixel_to_mm >= self.gripper_tip_thickness_mm / 2:
            available_width_mm = (extent_left + extent_right) * pixel_to_mm
        else:
            symmetric_extent_px = min(extent_left, extent_right)
            available_width_mm = symmetric_extent_px * 2 * pixel_to_mm

        # Check symmetry
        is_symmetric = abs(top_clearance_mm - bottom_clearance_mm) <= self.symmetry_tolerance_mm

        valid = (max_clearance_mm <= gripper_half_mm and
                 is_symmetric and
                 available_width_mm >= self.gripper_tip_thickness_mm)

        return valid, max_clearance_mm, blocking_regions

    def _find_tip_regions_x(
        self,
        region: Dict[str, Any],
        blocking_region_objs: List[Dict[str, Any]],
        all_regions: List[Dict[str, Any]],
        gripper_half_px: float,
        grasp_center_x: float,
        grasp_center_y: float
    ) -> List[Dict[str, Any]]:
        """Find all regions that provide tip support for X-axis grasp."""
        tip_regions = list(blocking_region_objs)
        tip_region_ids = {r['id'] for r in tip_regions}

        for blocking_region in blocking_region_objs:
            for other in all_regions:
                if other['id'] in tip_region_ids:
                    continue

                # Check if Y-adjacent (within MAX_GAP_PX)
                v_adjacent = (abs(blocking_region['bottom'] - other['top']) <= self.max_gap_px or
                             abs(other['bottom'] - blocking_region['top']) <= self.max_gap_px)

                if not v_adjacent:
                    continue

                # Check if maintains X-continuity
                x_overlap_with_blocking = not (other['right'] < blocking_region['left'] or
                                              other['left'] > blocking_region['right'])

                if not x_overlap_with_blocking:
                    continue

                # Verify it's within gripper's reach
                gripper_left = grasp_center_x - gripper_half_px
                gripper_right = grasp_center_x + gripper_half_px
                x_overlap_with_gripper = not (other['right'] < gripper_left or other['left'] > gripper_right)

                if not x_overlap_with_gripper:
                    continue

                # Check if accessible from X-axis direction
                if self._has_x_blocking(other, all_regions, tip_region_ids, grasp_center_x, gripper_half_px):
                    continue

                tip_regions.append(other)
                tip_region_ids.add(other['id'])

        return tip_regions

    def _find_tip_regions_y(
        self,
        region: Dict[str, Any],
        blocking_region_objs: List[Dict[str, Any]],
        all_regions: List[Dict[str, Any]],
        gripper_half_px: float,
        grasp_center_x: float,
        grasp_center_y: float
    ) -> List[Dict[str, Any]]:
        """Find all regions that provide tip support for Y-axis grasp."""
        tip_regions = list(blocking_region_objs)
        tip_region_ids = {r['id'] for r in tip_regions}

        for blocking_region in blocking_region_objs:
            for other in all_regions:
                if other['id'] in tip_region_ids:
                    continue

                # Check if X-adjacent (within MAX_GAP_PX)
                h_adjacent = (abs(blocking_region['right'] - other['left']) <= self.max_gap_px or
                             abs(other['right'] - blocking_region['left']) <= self.max_gap_px)

                if not h_adjacent:
                    continue

                # Check if maintains Y-continuity
                y_overlap_with_blocking = not (other['bottom'] < blocking_region['top'] or
                                              other['top'] > blocking_region['bottom'])

                if not y_overlap_with_blocking:
                    continue

                # Verify it's within gripper's reach
                gripper_top = grasp_center_y - gripper_half_px
                gripper_bottom = grasp_center_y + gripper_half_px
                y_overlap_with_gripper = not (other['bottom'] < gripper_top or other['top'] > gripper_bottom)

                if not y_overlap_with_gripper:
                    continue

                # Check if accessible from Y-axis direction
                if self._has_y_blocking(other, all_regions, tip_region_ids, grasp_center_y, gripper_half_px):
                    continue

                tip_regions.append(other)
                tip_region_ids.add(other['id'])

        return tip_regions

    def _has_x_blocking(
        self,
        other: Dict[str, Any],
        all_regions: List[Dict[str, Any]],
        tip_region_ids: set,
        grasp_center_x: float,
        gripper_half_px: float
    ) -> bool:
        """Check if a region has X-blocking obstacles."""
        other_center_x = (other['left'] + other['right']) / 2
        other_on_left = other_center_x < grasp_center_x
        other_on_right = other_center_x > grasp_center_x

        gripper_left = grasp_center_x - gripper_half_px
        gripper_right = grasp_center_x + gripper_half_px

        for obstacle in all_regions:
            if obstacle['id'] == other['id']:
                continue

            if obstacle['id'] in tip_region_ids:
                continue

            # Skip obstacles on the opposite side
            obstacle_center_x = (obstacle['left'] + obstacle['right']) / 2
            if other_on_left and obstacle_center_x > grasp_center_x:
                continue
            if other_on_right and obstacle_center_x < grasp_center_x:
                continue

            # Check Y overlap
            y_overlap = not (obstacle['bottom'] < other['top'] or obstacle['top'] > other['bottom'])
            if not y_overlap:
                continue

            # Check if obstacle is adjacent or intersecting
            x_intersect = (gripper_left < obstacle['right']) and (gripper_right > obstacle['left'])
            adjacent_left = (obstacle['right'] < other['left']) and (other['left'] - obstacle['right'] <= self.max_gap_px)
            adjacent_right = (obstacle['left'] > other['right']) and (obstacle['left'] - other['right'] <= self.max_gap_px)

            if x_intersect or adjacent_left or adjacent_right:
                return True

        return False

    def _has_y_blocking(
        self,
        other: Dict[str, Any],
        all_regions: List[Dict[str, Any]],
        tip_region_ids: set,
        grasp_center_y: float,
        gripper_half_px: float
    ) -> bool:
        """Check if a region has Y-blocking obstacles."""
        other_center_y = (other['top'] + other['bottom']) / 2
        other_on_top = other_center_y < grasp_center_y
        other_on_bottom = other_center_y > grasp_center_y

        gripper_top = grasp_center_y - gripper_half_px
        gripper_bottom = grasp_center_y + gripper_half_px

        for obstacle in all_regions:
            if obstacle['id'] == other['id']:
                continue

            if obstacle['id'] in tip_region_ids:
                continue

            # Skip obstacles on the opposite side
            obstacle_center_y = (obstacle['top'] + obstacle['bottom']) / 2
            if other_on_top and obstacle_center_y > grasp_center_y:
                continue
            if other_on_bottom and obstacle_center_y < grasp_center_y:
                continue

            # Check X overlap
            x_overlap = not (obstacle['right'] < other['left'] or obstacle['left'] > other['right'])
            if not x_overlap:
                continue

            # Check if obstacle is adjacent or intersecting
            y_intersect = (gripper_top < obstacle['bottom']) and (gripper_bottom > obstacle['top'])
            adjacent_top = (obstacle['bottom'] < other['top']) and (other['top'] - obstacle['bottom'] <= self.max_gap_px)
            adjacent_bottom = (obstacle['top'] > other['bottom']) and (obstacle['top'] - other['bottom'] <= self.max_gap_px)

            if y_intersect or adjacent_top or adjacent_bottom:
                return True

        return False

    def filter_grasp_points(
        self,
        grasp_data: Dict[str, Any],
        regions: List[Dict[str, Any]],
        check_x_axis: bool = True,
        check_y_axis: bool = True
    ) -> Dict[str, Any]:
        """
        Filter grasp points based on gripper constraints.

        Args:
            grasp_data: Grasp points data loaded from JSON file
            regions: List of detected regions from region detection
            check_x_axis: Whether to check X-axis grasps
            check_y_axis: Whether to check Y-axis grasps

        Returns:
            Dictionary with filtering results including valid grasp points
        """
        pixel_to_mm = self.compute_pixel_to_mm(grasp_data)
        results = []

        for region in regions:
            region_id = region['id']

            # Check X-axis grasps if enabled
            valid_x = []
            if check_x_axis:
                valid_x_half, _, _ = self.check_x_grasp(region, regions, pixel_to_mm, self.half_half_open_mm)
                valid_x_open, _, _ = self.check_x_grasp(region, regions, pixel_to_mm, self.half_open_mm)

                if valid_x_half:
                    valid_x.append('half-open')
                if valid_x_open:
                    valid_x.append('open')

            # Check Y-axis grasps if enabled
            valid_y = []
            if check_y_axis:
                valid_y_half, _, _ = self.check_y_grasp(region, regions, pixel_to_mm, self.half_half_open_mm)
                valid_y_open, _, _ = self.check_y_grasp(region, regions, pixel_to_mm, self.half_open_mm)

                if valid_y_half:
                    valid_y.append('half-open')
                if valid_y_open:
                    valid_y.append('open')

            results.append({
                'grasp_id': region_id,
                'valid_x': valid_x,
                'valid_y': valid_y,
                'is_valid': len(valid_x) > 0 or len(valid_y) > 0
            })

        # Filter grasp points to only include valid ones
        valid_grasp_ids = {r['grasp_id'] for r in results if r['is_valid']}

        filtered_grasp_points = [
            gp for gp in grasp_data['grasp_points']
            if gp['id'] in valid_grasp_ids
        ]

        return {
            'results': results,
            'filtered_grasp_points': filtered_grasp_points,
            'total_original': len(grasp_data['grasp_points']),
            'total_filtered': len(filtered_grasp_points),
            'filter_params': {
                'gripper_max_width_mm': self.gripper_max_width_mm,
                'gripper_half_open_width_mm': self.gripper_half_open_width_mm,
                'gripper_tip_thickness_mm': self.gripper_tip_thickness_mm,
                'max_gap_px': self.max_gap_px,
                'symmetry_tolerance_mm': self.symmetry_tolerance_mm
            }
        }
