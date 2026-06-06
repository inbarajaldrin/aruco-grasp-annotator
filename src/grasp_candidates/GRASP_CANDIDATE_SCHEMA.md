# Grasp Candidate — Canonical Schema & Math

**Status:** authoritative spec for `generate_grasp_candidates.py` and any node that publishes
oriented grasp candidates (e.g. the grasp-candidate publisher in `ros-mcp-server`).

This standardizes the grasp-candidate representation so that **its math provably reproduces the
already-verified executor target** (`move_to_grasp.py`'s `face_down(object_yaw)` + closing-axis
selection). The verification is numeric and reproducible (see §6); it is not asserted.

It is the pipeline-specific concretization of the general `(V, R, D, W)` grasp model documented in
`grasp_candidate_schema.md` (GraspNet-1Billion). Where that doc is generic, this one fixes the
gripper convention, the object→world composition, and the units used in this repo.

---

## 1. What a candidate is

A grasp candidate is a **full 6-DoF gripper-TCP target, expressed in the object's CAD-center frame**,
plus the actuator width. At runtime the live object pose maps it into the robot base frame.

It is described by four interpretable quantities (the source of truth) and one derived quaternion:

| Field | Symbol | Meaning |
|-------|--------|---------|
| `approach_vector` | `V` | unit 3-vector in object frame; direction the gripper advances toward the object. Flat-on-table objects use top-down `V = [0,0,1]`. |
| `in_plane_rotation_deg` | `R` | rotation about the approach axis selecting the **gripper closing axis**: `0°` → close along object **X**, `90°` → close along object **Y**. |
| `width_mm` | `W` | commanded jaw opening, taken from `grasp_validity` (`x_axis_gripper_width_mm` for `R=0`, `y_axis_gripper_width_mm` for `R=90`). |
| `standoff_m` | `D` | approach standoff applied along `V` at runtime (pre-grasp hover). Repo default `0.115`. |
| `approach_quaternion` | `q_cand` | **derived**: full gripper-TCP orientation in the object frame (includes the gripper mount flip). See §4. |

One grasp point yields **one candidate per non-null validity axis** (≤2): if
`x_axis_gripper_width_mm` is set → an `R=0` candidate; if `y_axis_gripper_width_mm` is set → an
`R=90` candidate. This is exactly the reachable, executor-matching regime for flat objects. The old
18-fixed-direction expansion (arbitrary in-plane yaw, no width) is superseded.

---

## 2. Frames & conventions

- **Object / CAD-center frame** — where grasp points and `grasp_validity` are defined
  (`coordinate_frame: "cad_center"`). Local axes X, Y (in the board plane), Z (board normal, "up").
- **Gripper TCP frame** — tool-center frame the arm commands. Face-down means TCP −Z points down onto
  the object's top (+Z) face.
- **Base / world frame** — robot base; object pose arrives here from `aruco_camera_localizer`
  (`/objects_poses_*`, a `TFMessage` of position + quaternion).

### The gripper mount flip `q_conv`

A fixed constant relating the object-frame approach to the physical face-down TCP:

```
q_conv = R.from_euler('xyz', [180, 0, 180], degrees=True)      # quaternion (x,y,z,w) = (0, 1, 0, 0)
```

It encodes "gripper approaches along its −Z onto the object +Z face." Derived by matching the
executor's `_face_down_quaternion`; it is **baked into `q_cand`** so downstream consumers need no
gripper knowledge.

---

## 3. The candidate orientation (object frame)

```
q_cand(V, R) = avq(V) · Rz(R) · q_conv
```

- `avq(V)` aligns the local approach to `V` (for top-down `V=[0,0,1]`, `avq = I`). This is the same
  `approach_vector_to_quaternion` already in the generator.
- `Rz(R)` is the in-plane rotation about the approach axis (`R ∈ {0°, 90°}`) that selects the closing
  axis.
- `q_conv` is the mount flip (§2).

Concrete top-down values (`avq = I`):

| `R` | closes along | `q_cand` (x,y,z,w) | `W` source |
|-----|--------------|--------------------|------------|
| `0°` | object **X** | `( 0.0000, 1.0000, 0, 0)` | `x_axis_gripper_width_mm` |
| `90°` | object **Y** | `(−0.7071, 0.7071, 0, 0)` | `y_axis_gripper_width_mm` |

---

## 4. Object → world (runtime projection)

Given the live object pose `(p_obj, q_obj)` in base frame:

```
q_world = q_obj ⊗ q_cand                                  # gripper TCP orientation in base
p_world = p_obj + R(q_obj) · p_candidate_object           # grasp point in base
p_pregrasp = p_world + D · ( R(q_world) · approach_axis )  # hover standoff, optional
```

`q_world` is directly the gripper command (face-down for top-down candidates) — **no downstream
face-down hack required.** This is what a candidate publisher emits per object, mirroring how
`grasp_points_publisher.py` already transforms position.

### 4.1 Yaw-collapse (canonical projection for the current executor)

The verified executor controls **yaw only** (objects lie flat, gripper always face-down;
`move_to_grasp` does `extract_yaw -> face_down`). Real detected poses carry small roll/pitch
*tilt noise* that the executor discards but a raw `q_obj ⊗ q_cand` would carry into the gripper.
So the publisher's canonical world projection is the **yaw-collapsed** form:

```
q_world = face_down( extract_yaw( q_obj ⊗ q_cand ) )
        = R.from_euler('xyz', [0, 180, yaw], degrees=True),   yaw = extract_yaw(q_obj ⊗ q_cand)
```

- For a **flat** object (`q_obj = Rz(θ)`) this is **identical** to `q_obj ⊗ q_cand` (proven, §6) —
  yaw-collapse changes nothing in the operating regime.
- For a **tilted** detection it absorbs the tilt, so the candidate path reproduces the executor
  **exactly on any pose**, robust to a few degrees of detection noise. This matters for consumers
  that command the published orientation directly (e.g. `visual_servo_grasp.py` uses the published
  RPY); `move_to_grasp` re-collapses anyway so it is immune either way.

The baked `approach_quaternion` stays the **full** TCP-in-object `q_cand` (general, 6-DoF-ready);
yaw-collapse is a *projection-time* policy of the current executor, not a property of the candidate.
A future 6-DoF executor would simply drop the collapse and consume `q_obj ⊗ q_cand` directly.

---

## 5. JSON schema (per object file `data/grasp_candidates/{object}_grasp_candidates.json`)

```json
{
  "object_name": "line_red",
  "schema_version": 2,
  "gripper": { "max_width_mm": 100, "clearance_mm": 14, "tip_thickness_mm": 20 },
  "total_grasp_candidates": 1,
  "grasp_candidates": [
    {
      "grasp_point_id": 1,
      "candidate_id": 1,
      "approach_name": "top",
      "approach_vector": { "x": 0.0, "y": 0.0, "z": 1.0 },
      "in_plane_rotation_deg": 0.0,
      "closing_axis": "x",
      "width_mm": 39.8,
      "standoff_m": 0.115,
      "grasp_candidate_position": { "x": -0.000106, "y": 0.000106, "z": 0.0 },
      "approach_quaternion": { "x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0 },
      "approach_rpy": { "roll": 180.0, "pitch": 0.0, "yaw": 180.0 }
    }
  ]
}
```

**Units:** positions/standoff in **meters**; width in **millimeters** (matches `gripper.json` and
`grasp_validity`); angles in **degrees**; quaternion `(x,y,z,w)`, scipy order.

Worked example above is the real `line_red` grasp point 1 (`pos ≈ (-1.06e-4, 1.06e-4, 0)`,
`x_axis_gripper_width_mm = 39.8`, `y = null`) → exactly **one** candidate (X-close, 39.8 mm).

---

## 6. Verification (the standard is *checkable*, not asserted)

For any candidate with object pose set to identity, composing must equal the executor's verified
target:

```
q_obj = Rz(object_yaw)
assert q_obj ⊗ q_cand(V,R)  ==  face_down(object_yaw + (90 if R==90 else 0))
   where face_down(a) = R.from_euler('xyz', [0, 180, a], degrees=True)
```

Confirmed for `object_yaw ∈ {0,30,90,137,-50}`, both `R∈{0,90}` → exact match (atol 1e-6). The
verified production path (`ENABLE_X_AXIS_GRASPS=True`, `ENABLE_Y_AXIS_GRASPS=False`) is the `R=0`
subset; `R=90` is the same math, currently gated off in `get_scene_info.py`.

**Regression rule:** regenerating candidates must not change `width_mm` (= `grasp_validity`) or, for
`R=0`, the `face_down` equality above. Diff against the previous run before committing.

---

## 7. Relation to `grasp_candidate_schema.md` (GraspNet)

| GraspNet `(V,R,D,W)` | Here |
|----------------------|------|
| approaching vector `V` | `approach_vector` (top-down for flat objects) |
| in-plane rotation `R` | `in_plane_rotation_deg` (0°/90° = closing-axis select) |
| approaching distance `D` | `standoff_m` (0.115 runtime offset) |
| gripper width `W` | `width_mm` (from `grasp_validity`, bounded by `gripper.json.max_width_mm`) |
| pose `T = [Rot·B \| t]` | `q_world = q_obj ⊗ q_cand`, `p_world = p_obj + R(q_obj)·p_cand` |

The generic doc's permutation `B` and the gripper convention are absorbed here into the single
constant `q_conv`.
