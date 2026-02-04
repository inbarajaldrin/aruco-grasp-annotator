"""Routes for Grasp Candidates Visualizer."""

from .data import router as data_router
from .robot import router as robot_router

__all__ = ["data_router", "robot_router"]
