"""Robot control routes for grasp execution and gripper commands."""

import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api")


class ExecuteGraspRequest(BaseModel):
    """Request model for grasp execution."""

    object_name: str
    grasp_id: str  # Format: "grasp_point_id_direction_id" or legacy integer
    topic: str = "/objects_poses_sim"
    movement_duration: float = 10.0
    grasp_point_id: Optional[int] = None
    direction_id: Optional[int] = None


class GripperCommandRequest(BaseModel):
    """Request model for gripper commands."""

    command: str  # 'open' or 'close'


def get_script_dir() -> Path:
    """Get the script directory path."""
    return Path(__file__).parent.parent


def log_process_output(pipe, process_pid: int) -> None:
    """Read output from process and log it."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                line = line.strip()
                if line:
                    print(f"[PID {process_pid}] {line}")
        pipe.close()
    except Exception as e:
        print(f"Error reading process output: {e}")


def start_background_process(cmd_str: str, cwd: Path):
    """Start a background process and return it with logging thread."""
    print(f"Executing command: {cmd_str}")
    process = subprocess.Popen(
        cmd_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd),
        shell=True,
        executable="/bin/bash",
        text=True,
        bufsize=1,
    )

    output_thread = threading.Thread(
        target=log_process_output, args=(process.stdout, process.pid), daemon=True
    )
    output_thread.start()

    time.sleep(0.5)

    return_code = process.poll()
    if return_code is not None:
        error_output = ""
        try:
            if process.stdout:
                remaining = process.stdout.read()
                if remaining:
                    error_output = remaining.strip()
        except Exception:
            pass

        error_msg = f"Process exited immediately with code {return_code}"
        if error_output:
            error_msg += f". Output: {error_output[:500]}"
        print(f"Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"Process started successfully with PID: {process.pid}")
    return process


def parse_grasp_id(
    grasp_id: str, grasp_point_id: Optional[int], direction_id: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """Parse grasp_id into grasp_point_id and direction_id."""
    if grasp_point_id is not None and direction_id is not None:
        return grasp_point_id, direction_id

    if not grasp_id:
        raise HTTPException(
            status_code=400,
            detail="grasp_id or both grasp_point_id and direction_id are required",
        )

    if "_" in str(grasp_id):
        parts = str(grasp_id).split("_")
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass

    # Try legacy integer format
    try:
        int(grasp_id)
        return None, None  # Legacy mode
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid grasp_id format: {grasp_id}. Expected 'grasp_point_id_direction_id' or integer",
        )


@router.post("/execute-grasp")
async def execute_grasp(request: ExecuteGraspRequest):
    """Execute visual servo grasp for the specified object and grasp candidate."""
    grasp_point_id, direction_id = parse_grasp_id(
        request.grasp_id, request.grasp_point_id, request.direction_id
    )

    if not request.object_name:
        raise HTTPException(status_code=400, detail="object_name is required")

    script_dir = get_script_dir()
    script_path = script_dir / "visual_servo_grasp.py"

    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")

    # Remove _scaled70 suffix for topic object name
    topic_object_name = request.object_name.replace("_scaled70", "")

    cmd_parts = [
        "source /opt/ros/humble/setup.bash &&",
        f"python3.10 {script_path}",
        f"--object-name {topic_object_name}",
        f"--topic {request.topic}",
        f"--movement-duration {request.movement_duration}",
    ]

    if grasp_point_id is not None and direction_id is not None:
        cmd_parts.extend(
            [f"--grasp-point-id {grasp_point_id}", f"--direction-id {direction_id}"]
        )
    else:
        cmd_parts.append(f"--grasp-id {request.grasp_id}")

    cmd_str = " ".join(cmd_parts)
    process = start_background_process(cmd_str, script_dir.parent.parent)

    if grasp_point_id is not None and direction_id is not None:
        message = f"Executing grasp for {topic_object_name} at grasp_point_id {grasp_point_id}, direction_id {direction_id}"
    else:
        message = f"Executing grasp for {topic_object_name} at grasp point {request.grasp_id} (legacy mode)"

    return JSONResponse(
        content={
            "status": "started",
            "message": message,
            "pid": process.pid,
            "grasp_point_id": grasp_point_id,
            "direction_id": direction_id,
        }
    )


@router.post("/gripper-command")
async def gripper_command(request: GripperCommandRequest):
    """Send gripper command (open/close) via ROS2 topic."""
    command = request.command.lower()

    if command not in ["open", "close"]:
        raise HTTPException(status_code=400, detail="Command must be 'open' or 'close'")

    cmd_str = (
        f"source /opt/ros/humble/setup.bash && "
        f"ros2 topic pub --once /gripper_command std_msgs/String \"{{data: '{command}'}}\""
    )

    print(f"Executing gripper command: {cmd_str}")
    try:
        result = subprocess.run(
            cmd_str,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=5.0,
        )

        if result.returncode != 0:
            error_msg = f"Gripper command failed with code {result.returncode}"
            if result.stderr:
                error_msg += f". Error: {result.stderr[:200]}"
            print(f"Error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

        print(f"Gripper command '{command}' sent successfully")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}")

        return JSONResponse(
            content={
                "status": "success",
                "message": f"Gripper {command} command sent successfully",
                "command": command,
            }
        )

    except subprocess.TimeoutExpired:
        error_msg = "Gripper command timed out"
        print(f"Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error sending gripper command: {str(e)}"
        print(f"Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/move-to-safe-height")
async def move_to_safe_height():
    """Execute move_to_safe_height.py script."""
    script_dir = get_script_dir()
    script_path = script_dir / "move_to_safe_height.py"

    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")

    cmd_str = f"source /opt/ros/humble/setup.bash && python3.10 {script_path}"
    process = start_background_process(cmd_str, script_dir.parent.parent)

    return JSONResponse(
        content={
            "status": "started",
            "message": "Moving to safe height",
            "pid": process.pid,
        }
    )


@router.post("/move-home")
async def move_home():
    """Execute move_home.py script."""
    script_dir = get_script_dir()
    script_path = script_dir / "move_home.py"

    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path}")

    cmd_str = f"source /opt/ros/humble/setup.bash && python3.10 {script_path}"
    process = start_background_process(cmd_str, script_dir.parent.parent)

    return JSONResponse(
        content={
            "status": "started",
            "message": "Moving to home position",
            "pid": process.pid,
        }
    )
