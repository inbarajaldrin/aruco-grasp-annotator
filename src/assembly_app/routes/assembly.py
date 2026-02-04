"""Assembly management routes for Assembly App."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api")

# Global assembly state
assembly_state = {
    "components": {},
    "assemblies": [],
    "next_id": 1,
}


@router.post("/assembly")
async def save_assembly(assembly_data: dict):
    """Save assembly configuration."""
    assembly_id = assembly_state["next_id"]
    assembly_state["next_id"] += 1

    assembly_state["assemblies"].append(
        {
            "id": assembly_id,
            "data": assembly_data,
            "timestamp": "2024-01-01T12:00:00",
        }
    )

    return JSONResponse({"assembly_id": assembly_id, "success": True})


@router.get("/assemblies")
async def get_assemblies():
    """Get all saved assemblies."""
    return assembly_state["assemblies"]
