"""Extract and verify the exact planar contact kinematics of dm_control cheetah.

The structured port-Hamiltonian model (models/port_hamiltonian.py) currently
learns its contact geometry with a gap MLP.  That is unnecessary: the floor is a
plane at z = 0 and the signed gap of every collidable capsule endpoint is exact
forward kinematics of the observed generalized positions.  This script walks
mjModel programmatically, derives a closed-form planar chain spec for every
capsule endpoint, implements the batched torch FK + analytic Jacobian from that
spec, and verifies all of it against MuJoCo.

Nothing here is transcribed by hand: every constant is read out of mjModel and
the hand-checked reference values are only used as assertions.

Run:
    MUJOCO_GL=disable python evaluations/extract_contact_kinematics.py
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "disable")

import numpy as np
import mujoco
import torch
from dm_control import suite


# --------------------------------------------------------------------------
# 1. mjModel walk + planarity preconditions
# --------------------------------------------------------------------------

PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)
CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
JNT_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)

# Planar convention: rotation about +y by theta maps (x, z) -> (x c + z s, -x s + z c).
AXIS_Y = np.array([0.0, 1.0, 0.0])
TOL = 1e-12


def _name(model, objtype, i):
    return mujoco.mj_id2name(model, objtype, i)


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.ascontiguousarray(quat, dtype=float))
    return mat.reshape(3, 3)


class PlanarityError(AssertionError):
    pass


def check_planarity(model) -> Dict[str, object]:
    """Assert every precondition the closed form depends on.  Loud on failure."""
    problems: List[str] = []

    # --- the floor -------------------------------------------------------
    planes = [g for g in range(model.ngeom) if model.geom_type[g] == PLANE]
    if len(planes) != 1:
        problems.append(f"expected exactly 1 plane geom, found {len(planes)}")
    floor = planes[0] if planes else None
    if floor is not None:
        if abs(model.geom_pos[floor][2]) > TOL:
            problems.append(f"floor plane z = {model.geom_pos[floor][2]!r}, expected 0")
        fmat = _quat_to_mat(model.geom_quat[floor])
        normal = fmat[:, 2]
        if not np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-12):
            problems.append(f"floor normal is {normal}, expected +z")
        if model.geom_bodyid[floor] != 0:
            problems.append("floor is not attached to the world body")

    # --- joints ----------------------------------------------------------
    for j in range(model.njnt):
        jn = _name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jt = int(model.jnt_type[j])
        if jt not in (JNT_HINGE, JNT_SLIDE):
            problems.append(f"joint {j} ({jn}) type {jt} is neither hinge nor slide")
            continue
        if np.abs(model.jnt_pos[j]).max() > TOL:
            problems.append(f"joint {j} ({jn}) jnt_pos = {model.jnt_pos[j]} != 0")
        ax = np.asarray(model.jnt_axis[j], dtype=float)
        if jt == JNT_HINGE:
            if not (np.allclose(ax, AXIS_Y, atol=1e-12) or np.allclose(ax, -AXIS_Y, atol=1e-12)):
                problems.append(f"hinge {j} ({jn}) axis {ax} is not +/-(0,1,0)")
        else:  # slide
            if abs(ax[1]) > TOL:
                problems.append(f"slide {j} ({jn}) axis {ax} has a y component")

    # --- body frames -----------------------------------------------------
    for b in range(1, model.nbody):
        bn = _name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if abs(model.body_pos[b][1]) > TOL:
            problems.append(f"body {b} ({bn}) body_pos has a y offset {model.body_pos[b]}")
        q = np.asarray(model.body_quat[b], dtype=float)
        if abs(q[1]) > TOL or abs(q[3]) > TOL:
            problems.append(f"body {b} ({bn}) body_quat {q} is not a pure-y rotation")

    # --- collidable geoms ------------------------------------------------
    collidable = []
    for g in range(model.ngeom):
        if g == floor:
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        gn = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        collidable.append(g)
        if int(model.geom_type[g]) != CAPSULE:
            problems.append(f"geom {g} ({gn}) type {model.geom_type[g]} is not a capsule")
        q = np.asarray(model.geom_quat[g], dtype=float)
        if abs(q[1]) > TOL or abs(q[3]) > TOL:
            problems.append(f"geom {g} ({gn}) quat {q} is not a pure-y rotation")
        if abs(model.geom_pos[g][1]) > TOL:
            problems.append(f"geom {g} ({gn}) geom_pos has a y offset {model.geom_pos[g]}")
        if model.geom_margin[g] != 0.0 or model.geom_gap[g] != 0.0:
            problems.append(
                f"geom {g} ({gn}) has margin={model.geom_margin[g]} gap={model.geom_gap[g]}"
            )

    if problems:
        raise PlanarityError(
            "PLANARITY PRECONDITIONS FAILED -- the closed-form contact kinematics "
            "are NOT valid for this model:\n  " + "\n  ".join(problems)
        )

    return {
        "floor_geom": int(floor),
        "floor_geom_name": _name(model, mujoco.mjtObj.mjOBJ_GEOM, floor),
        "floor_plane_z": float(model.geom_pos[floor][2]),
        "collidable_geoms": [int(g) for g in collidable],
    }


# --------------------------------------------------------------------------
# 2. chain-spec extraction
# --------------------------------------------------------------------------


@dataclass
class EndpointSpec:
    """One capsule endpoint's planar kinematic chain.

    z(pos) = base_z + pos[root_z_pos_index] + sum_k cz_k
    x(pos) = x_root + base_x + sum_k cx_k
    with Theta_k = sum_{j<=k} sign_j * pos[angle_pos_indices[j]] and
      cx_k =  dx_k cos(Theta_k) + dz_k sin(Theta_k)
      cz_k = -dx_k sin(Theta_k) + dz_k cos(Theta_k)
    gap(pos) = z(pos) - radius.
    """

    name: str
    geom: str
    geom_id: int
    end: str  # "plus" | "minus"
    radius: float
    half_length: float
    angle_pos_indices: List[int]
    angle_signs: List[float]
    offsets: List[List[float]]  # (dx, dz) per segment, same length as angles


def _joints_of_body(model, b: int) -> List[int]:
    adr = int(model.body_jntadr[b])
    num = int(model.body_jntnum[b])
    return list(range(adr, adr + num)) if num > 0 else []


def build_specs(model) -> Tuple[List[EndpointSpec], Dict[str, object]]:
    """Walk the tree and emit a chain spec for every collidable capsule endpoint.

    Observed-position index space: obs pos = qpos[1:nq], so pos index = qposadr - 1.
    """
    info = check_planarity(model)
    floor = info["floor_geom"]

    # The root slide DOFs.  They must precede every hinge in their body's joint
    # list (otherwise their axes would be rotated and the base offset would not
    # be constant), and their axes must be world x / world z.
    root_z_pos_index = None
    root_x_cfg = None
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != JNT_SLIDE:
            continue
        b = int(model.jnt_bodyid[j])
        if int(model.body_parentid[b]) != 0:
            raise PlanarityError(f"slide joint {j} is not on a root body")
        for k in _joints_of_body(model, b):
            if k < j and int(model.jnt_type[k]) == JNT_HINGE:
                raise PlanarityError(
                    f"slide joint {j} follows hinge {k} on the same body; its axis "
                    "would be rotated and the base offset would not be constant"
                )
        ax = np.asarray(model.jnt_axis[j], dtype=float)
        adr = int(model.jnt_qposadr[j])
        if np.allclose(ax, [0, 0, 1], atol=1e-12):
            root_z_pos_index = adr - 1
        elif np.allclose(ax, [1, 0, 0], atol=1e-12):
            root_x_cfg = adr
        else:
            raise PlanarityError(f"slide joint {j} axis {ax} is neither world x nor world z")
    if root_z_pos_index is None:
        raise PlanarityError("no world-z slide joint found; the gap would be unobservable")

    # Walk root -> body for each body once, caching the per-body chain.
    # chain[b] = (angle_pos_indices, angle_signs, offsets, base_xz)
    chain_cache: Dict[int, Tuple[List[int], List[float], List[np.ndarray], np.ndarray]] = {}

    def chain_for_body(b: int):
        if b in chain_cache:
            return chain_cache[b]
        if b == 0:
            chain_cache[b] = ([], [], [], np.zeros(2))
            return chain_cache[b]
        p = int(model.body_parentid[b])
        angs, signs, offs, base = chain_for_body(p)
        angs, signs, offs, base = list(angs), list(signs), [o.copy() for o in offs], base.copy()

        # body_pos is expressed in the parent frame == the frame after the last
        # hinge already in the chain.
        bp = np.array([model.body_pos[b][0], model.body_pos[b][2]])
        if offs:
            offs[-1] = offs[-1] + bp
        else:
            base = base + bp

        # body_quat is a constant pure-y rotation; fold it into the chain by
        # opening a segment with a constant angle if it is not identity.
        bq = np.asarray(model.body_quat[b], dtype=float)
        if not np.allclose(bq, [1.0, 0.0, 0.0, 0.0], atol=1e-12):
            raise PlanarityError(
                f"body {b} has a non-identity body_quat {bq}; supported in principle "
                "but not implemented -- the spec format carries no constant angles"
            )

        for j in _joints_of_body(model, b):
            jt = int(model.jnt_type[j])
            if jt == JNT_SLIDE:
                continue  # handled as the root base offset / cyclic root x
            sgn = 1.0 if model.jnt_axis[j][1] > 0 else -1.0
            angs.append(int(model.jnt_qposadr[j]) - 1)
            signs.append(sgn)
            offs.append(np.zeros(2))

        chain_cache[b] = (angs, signs, offs, base)
        return chain_cache[b]

    specs: List[EndpointSpec] = []
    for g in info["collidable_geoms"]:
        b = int(model.geom_bodyid[g])
        angs, signs, offs, base = chain_for_body(b)
        gname = _name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        radius = float(model.geom_size[g][0])
        half = float(model.geom_size[g][1])
        gpos = np.array([model.geom_pos[g][0], model.geom_pos[g][2]])
        gmat = _quat_to_mat(model.geom_quat[g])
        axis = gmat[:, 2]  # capsule axis is its local z
        if abs(axis[1]) > 1e-12:
            raise PlanarityError(f"geom {g} ({gname}) capsule axis {axis} leaves the xz plane")
        axis_xz = np.array([axis[0], axis[2]])
        for end, sgn in (("plus", +1.0), ("minus", -1.0)):
            e_offs = [o.copy() for o in offs]
            e_base = base.copy()
            tip = gpos + sgn * half * axis_xz
            if e_offs:
                e_offs[-1] = e_offs[-1] + tip
            else:
                e_base = e_base + tip
            if not e_offs:
                raise PlanarityError(
                    f"geom {g} ({gname}) is rigidly attached to the world; no hinge in its chain"
                )
            specs.append(
                EndpointSpec(
                    name=f"{gname}_{end}",
                    geom=gname,
                    geom_id=int(g),
                    end=end,
                    radius=radius,
                    half_length=half,
                    angle_pos_indices=[int(a) for a in angs],
                    angle_signs=[float(s) for s in signs],
                    offsets=[[float(o[0]), float(o[1])] for o in e_offs],
                )
            )

    meta = {
        "env": "dm_control suite cheetah-run",
        "root_z_pos_index": int(root_z_pos_index),
        "root_x_cfg_index": int(root_x_cfg) if root_x_cfg is not None else None,
        "base_z_const": None,  # filled per-spec below (same for all here)
        "n_pos": int(model.nq - 1),
        "pos_index_names": [
            _name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(model.njnt)
            if int(model.jnt_qposadr[j]) >= 1
        ],
        **info,
    }
    # base offset is chain-independent (world -> first hinge); record it once.
    bases = {tuple(np.round(chain_for_body(int(model.geom_bodyid[g]))[3], 15))
             for g in info["collidable_geoms"]}
    if len(bases) != 1:
        raise PlanarityError(f"multiple distinct base offsets {bases}; spec assumes one")
    base_xz = list(bases)[0]
    meta["base_x_const"] = float(base_xz[0])
    meta["base_z_const"] = float(base_xz[1])
    return specs, meta


# --------------------------------------------------------------------------
# 3. batched torch FK + analytic Jacobian
# --------------------------------------------------------------------------


class PlanarContactKinematics:
    """Closed-form batched FK for every capsule endpoint over a z=0 floor.

    No autograd, no functorch: cumsum forward, flipped cumsum backward.
    """

    def __init__(self, specs: Sequence[EndpointSpec], meta: Dict[str, object],
                 n_pos: int, dtype=torch.float64):
        self.specs = list(specs)
        self.n_pos = int(n_pos)
        self.root_z = int(meta["root_z_pos_index"])
        self.base_z = float(meta["base_z_const"])
        self.base_x = float(meta["base_x_const"])
        self.K = len(self.specs)
        self.S = max(len(s.angle_pos_indices) for s in self.specs)

        # Pad every chain to the same depth S with a zero-offset segment whose
        # angle index points at a dummy slot (a zero column appended to pos).
        idx = torch.full((self.K, self.S), self.n_pos, dtype=torch.long)
        sgn = torch.zeros(self.K, self.S, dtype=dtype)
        dxz = torch.zeros(self.K, self.S, 2, dtype=dtype)
        for k, s in enumerate(self.specs):
            n = len(s.angle_pos_indices)
            # Right-align is unnecessary; left-align and let the padded tail
            # contribute Theta increments of 0 and offsets of 0.
            idx[k, :n] = torch.tensor(s.angle_pos_indices, dtype=torch.long)
            sgn[k, :n] = torch.tensor(s.angle_signs, dtype=dtype)
            dxz[k, :n] = torch.tensor(s.offsets, dtype=dtype)
        self.idx = idx
        self.sgn = sgn
        self.dxz = dxz
        self.radius = torch.tensor([s.radius for s in self.specs], dtype=dtype)
        self.dtype = dtype

    def _segments(self, pos: torch.Tensor):
        """pos: (B, n_pos) -> cx, cz each (B, K, S)."""
        B = pos.shape[0]
        padded = torch.cat([pos, pos.new_zeros(B, 1)], dim=1)          # (B, n_pos+1)
        theta_step = padded[:, self.idx.reshape(-1)].reshape(B, self.K, self.S)
        theta_step = theta_step * self.sgn                              # (B,K,S)
        theta = torch.cumsum(theta_step, dim=2)
        c, s = torch.cos(theta), torch.sin(theta)
        dx = self.dxz[..., 0]
        dz = self.dxz[..., 1]
        cx = dx * c + dz * s
        cz = -dx * s + dz * c
        return cx, cz

    def forward(self, pos: torch.Tensor):
        """pos: (B, n_pos) -> (x_rel, z) each (B, K).

        x_rel omits the unobserved root-x translation (it cancels in the gap and
        only shifts the tangent direction, which is the root-x onehot).
        """
        cx, cz = self._segments(pos)
        z = self.base_z + pos[:, self.root_z : self.root_z + 1] + cz.sum(dim=2)
        x = self.base_x + cx.sum(dim=2)
        return x, z

    def gap(self, pos: torch.Tensor) -> torch.Tensor:
        _, z = self.forward(pos)
        return z - self.radius

    def jacobians(self, pos: torch.Tensor):
        """Analytic dz/dpos and dx/dpos, each (B, K, n_pos)."""
        B = pos.shape[0]
        cx, cz = self._segments(pos)
        # reverse cumulative sums over the segment axis
        tail_x = torch.flip(torch.cumsum(torch.flip(cx, [2]), dim=2), [2])  # sum_{k>=j} cx_k
        tail_z = torch.flip(torch.cumsum(torch.flip(cz, [2]), dim=2), [2])
        dz_seg = -tail_x * self.sgn
        dx_seg = tail_z * self.sgn
        dz = pos.new_zeros(B, self.K, self.n_pos + 1)
        dx = pos.new_zeros(B, self.K, self.n_pos + 1)
        idx = self.idx.unsqueeze(0).expand(B, -1, -1)
        dz.scatter_add_(2, idx, dz_seg)
        dx.scatter_add_(2, idx, dx_seg)
        dz = dz[:, :, : self.n_pos]
        dx = dx[:, :, : self.n_pos]
        dz[:, :, self.root_z] = dz[:, :, self.root_z] + 1.0
        return dx, dz

    def segment_points(self, pos: torch.Tensor):
        """World (x_rel, z) of every intermediate hinge frame origin, (B,K,S,2).

        Point j is the origin of the frame in which offset d_j lives, i.e. the
        anchor of hinge a_j.  Used for the dz/dtheta_j == -(x_pt - x_j) check.
        """
        cx, cz = self._segments(pos)
        B = pos.shape[0]
        zero = cx.new_zeros(B, self.K, 1)
        px = self.base_x + torch.cat([zero, torch.cumsum(cx, dim=2)[:, :, :-1]], dim=2)
        pz = (
            self.base_z
            + pos[:, self.root_z].reshape(B, 1, 1)
            + torch.cat([zero, torch.cumsum(cz, dim=2)[:, :, :-1]], dim=2)
        )
        return px, pz


# --------------------------------------------------------------------------
# 4. verification
# --------------------------------------------------------------------------


def sample_states(model, n: int, rng: np.random.Generator) -> np.ndarray:
    """Broad qpos sampling: upright, upside down, airborne and deeply penetrating."""
    qpos = np.zeros((n, model.nq))
    qpos[:, 0] = rng.uniform(-2.0, 20.0, size=n)          # root x
    qpos[:, 1] = rng.uniform(-1.0, 2.0, size=n)           # root z (deep penetration -> flight)
    qpos[:, 2] = rng.uniform(-np.pi, np.pi, size=n)       # root pitch, incl. upside down
    for j in range(3, model.njnt):
        adr = int(model.jnt_qposadr[j])
        lo, hi = model.jnt_range[j]
        if not model.jnt_limited[j]:
            lo, hi = -np.pi, np.pi
        pad = 0.25 * (hi - lo)                            # 25% outside the rails too
        qpos[:, adr] = rng.uniform(lo - pad, hi + pad, size=n)
    return qpos


def mujoco_endpoints(model, data, qpos: np.ndarray, specs: Sequence[EndpointSpec]) -> np.ndarray:
    """(B, K, 2) world (x, z) of each endpoint straight out of mj_kinematics."""
    out = np.zeros((qpos.shape[0], len(specs), 2))
    for i in range(qpos.shape[0]):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        for k, s in enumerate(specs):
            g = s.geom_id
            axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
            sgn = 1.0 if s.end == "plus" else -1.0
            p = data.geom_xpos[g] + sgn * s.half_length * axis
            out[i, k, 0] = p[0]
            out[i, k, 1] = p[2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=6000)
    ap.add_argument("--n-jac", type=int, default=64, help="states for the autograd Jacobian check")
    ap.add_argument("--n-contact-rollout", type=int, default=400)
    ap.add_argument("--out", default="results/contact_kinematics/cheetah_contact_kinematics.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(args.seed)

    env = suite.load("cheetah", "run", task_kwargs={"random": args.seed})
    model = env.physics.model.ptr
    data = mujoco.MjData(model)

    specs, meta = build_specs(model)
    n_pos = model.nq - 1
    kin = PlanarContactKinematics(specs, meta, n_pos=n_pos)

    print(f"[spec] {len(specs)} endpoints, max chain depth {kin.S}, n_pos {n_pos}")
    print(f"[spec] base (x,z) = ({meta['base_x_const']}, {meta['base_z_const']}), "
          f"root_z pos index {meta['root_z_pos_index']}")
    for s in specs:
        print(f"  {s.name:14s} r={s.radius:.3f} angles={s.angle_pos_indices} "
              f"offsets={[[round(v, 6) for v in o] for o in s.offsets]}")

    # --- reference-value assertions (derived numbers must match the audit) ---
    ref = {
        "torso": [1], "head": [1], "bthigh": [1, 2], "bshin": [1, 2, 3], "bfoot": [1, 2, 3, 4],
        "fthigh": [1, 5], "fshin": [1, 5, 6], "ffoot": [1, 5, 6, 7],
    }
    for s in specs:
        assert s.angle_pos_indices == ref[s.geom], (s.name, s.angle_pos_indices)
        assert all(sg == 1.0 for sg in s.angle_signs)
    assert abs(meta["base_z_const"] - 0.7) < 1e-12
    print("[spec] derived chains match the independently audited reference")

    # --- (a) FK vs MuJoCo --------------------------------------------------
    qpos = sample_states(model, args.n_states, rng)
    pos_t = torch.tensor(qpos[:, 1:], dtype=torch.float64)
    x_t, z_t = kin.forward(pos_t)
    ref_xz = mujoco_endpoints(model, data, qpos, specs)
    # x is only determined up to the unobserved root translation
    x_full = x_t.numpy() + qpos[:, 0:1]
    err_x = np.abs(x_full - ref_xz[:, :, 0]).max()
    err_z = np.abs(z_t.numpy() - ref_xz[:, :, 1]).max()
    fk_err = float(max(err_x, err_z))
    print(f"[a] FK vs mj_kinematics over {args.n_states} states: "
          f"max|dx| = {err_x:.3e} m, max|dz| = {err_z:.3e} m")

    # --- (b) analytic Jacobian vs autograd ---------------------------------
    sub = torch.tensor(qpos[: args.n_jac, 1:], dtype=torch.float64)
    dx_a, dz_a = kin.jacobians(sub)

    def fk_z(p):
        return kin.forward(p.unsqueeze(0))[1].squeeze(0)

    def fk_x(p):
        return kin.forward(p.unsqueeze(0))[0].squeeze(0)

    ez = 0.0
    ex = 0.0
    for i in range(sub.shape[0]):
        jz = torch.autograd.functional.jacobian(fk_z, sub[i], vectorize=True)
        jx = torch.autograd.functional.jacobian(fk_x, sub[i], vectorize=True)
        ez = max(ez, float((jz - dz_a[i]).abs().max()))
        ex = max(ex, float((jx - dx_a[i]).abs().max()))
    jac_err = float(max(ez, ex))
    print(f"[b] analytic vs autograd Jacobian over {sub.shape[0]} states: "
          f"max|dz/dq err| = {ez:.3e}, max|dx/dq err| = {ex:.3e}")

    # also: MuJoCo's own point Jacobian (jacp) as a third opinion
    mj_jac_err = 0.0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for i in range(sub.shape[0]):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        for k, s in enumerate(specs):
            g = s.geom_id
            axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
            sgn = 1.0 if s.end == "plus" else -1.0
            p = data.geom_xpos[g] + sgn * s.half_length * axis
            mujoco.mj_jac(model, data, jacp, jacr, p, int(model.geom_bodyid[g]))
            # config DOF d>=1 maps to pos index d-1
            mj_jac_err = max(mj_jac_err, float(np.abs(jacp[2, 1:] - dz_a[i, k].numpy()).max()))
            mj_jac_err = max(mj_jac_err, float(np.abs(jacp[0, 1:] - dx_a[i, k].numpy()).max()))
            mj_jac_err = max(mj_jac_err, abs(jacp[2, 0]))          # dz/d(root x) == 0
            mj_jac_err = max(mj_jac_err, abs(jacp[0, 0] - 1.0))    # dx/d(root x) == 1
    print(f"[b'] analytic vs mj_jac point Jacobian: max abs err = {mj_jac_err:.3e}")

    # --- (4) the planar cross-product identity dz/dtheta_j == -(x_pt - x_j) --
    px, pz = kin.segment_points(sub)
    ident_z = 0.0
    ident_x = 0.0
    xs, zs = kin.forward(sub)
    for k, s in enumerate(specs):
        n = len(s.angle_pos_indices)
        for j in range(n):
            a = s.angle_pos_indices[j]
            ident_z = max(ident_z, float((dz_a[:, k, a] + (xs[:, k] - px[:, k, j])).abs().max()))
            ident_x = max(ident_x, float((dx_a[:, k, a] - (zs[:, k] - pz[:, k, j])).abs().max()))
    print(f"[4] cross-product identity: max|dz/dth_j + (x-x_j)| = {ident_z:.3e}, "
          f"max|dx/dth_j - (z-z_j)| = {ident_x:.3e}")

    # --- (c) gap vs MuJoCo contact distances --------------------------------
    floor = meta["floor_geom"]
    ends_of_geom: Dict[int, List[int]] = {}
    for k, s in enumerate(specs):
        ends_of_geom.setdefault(s.geom_id, []).append(k)

    # MuJoCo's capsule-plane collider can emit up to two contacts per capsule,
    # one per cap sphere, each with dist = (that endpoint height) - radius.  So
    # check both readings: every reported contact must match SOME endpoint of
    # that geom, and the per-geom minimum contact dist must equal our minimum
    # endpoint gap.
    per_contact_errs: List[float] = []
    per_geom_min_errs: List[float] = []
    n_contacts = 0
    n_pen = 0
    n_geom_states = 0

    def _check_state(d):
        nonlocal n_contacts, n_pen, n_geom_states
        p = torch.tensor(np.array(d.qpos)[1:], dtype=torch.float64).unsqueeze(0)
        gaps = kin.gap(p).squeeze(0).numpy()
        by_geom: Dict[int, List[float]] = {}
        for ci in range(d.ncon):
            con = d.contact[ci]
            g1, g2 = int(con.geom1), int(con.geom2)
            if floor not in (g1, g2):
                continue
            other = g2 if g1 == floor else g1
            if other not in ends_of_geom:
                continue
            dist = float(con.dist)
            n_contacts += 1
            n_pen += int(dist < 0)
            per_contact_errs.append(min(abs(gaps[k] - dist) for k in ends_of_geom[other]))
            by_geom.setdefault(other, []).append(dist)
        for g, dists in by_geom.items():
            n_geom_states += 1
            per_geom_min_errs.append(
                abs(min(dists) - min(gaps[k] for k in ends_of_geom[g]))
            )

    env.reset()
    spec_action = env.action_spec()
    for _ in range(args.n_contact_rollout):
        env.step(rng.uniform(spec_action.minimum, spec_action.maximum))
        _check_state(env.physics.data)

    # plus randomly posed states (many of them interpenetrating / upside down)
    qc = sample_states(model, args.n_contact_rollout, rng)
    qc[:, 1] = rng.uniform(-0.6, 0.3, size=qc.shape[0])  # force the floor band
    qc[:, 0] = rng.uniform(0.0, 5.0, size=qc.shape[0])   # stay inside the plane's extent
    for i in range(qc.shape[0]):
        data.qpos[:] = qc[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        _check_state(data)

    gap_err = float(max(per_contact_errs)) if per_contact_errs else float("nan")
    gap_min_err = float(max(per_geom_min_errs)) if per_geom_min_errs else float("nan")
    print(f"[c] {n_contacts} floor contacts ({n_pen} penetrating), "
          f"{n_geom_states} (state, geom) pairs")
    print(f"[c] max|contact.dist - nearest endpoint gap| = {gap_err:.3e} m")
    print(f"[c] max|min_i contact.dist - min_e endpoint gap| = {gap_min_err:.3e} m")

    # --- write the spec -----------------------------------------------------
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "meta": meta,
        "convention": {
            "pos": "obs[0:8] = qpos[1:9] = [rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot]",
            "theta_k": "sum_{j<=k} angle_signs[j] * pos[angle_pos_indices[j]]",
            "cx_k": "dx_k*cos(theta_k) + dz_k*sin(theta_k)",
            "cz_k": "-dx_k*sin(theta_k) + dz_k*cos(theta_k)",
            "z": "base_z_const + pos[root_z_pos_index] + sum_k cz_k",
            "x": "x_root + base_x_const + sum_k cx_k",
            "gap": "z - radius",
            "dz_dpos_rootz": "1",
            "dz_dpos_a_j": "-sign_j * sum_{k>=j} cx_k  == -(x_point - x_j)",
            "dx_dpos_a_j": "+sign_j * sum_{k>=j} cz_k  == +(z_point - z_j)",
            "dx_droot_x": "1 (config DOF 0, the friction tangent onehot)",
        },
        "endpoints": [asdict(s) for s in specs],
        "verification": {
            "n_states_checked": int(args.n_states),
            "fk_max_abs_error_m": fk_err,
            "fk_max_abs_error_x_m": float(err_x),
            "fk_max_abs_error_z_m": float(err_z),
            "jac_max_abs_error_autograd": jac_err,
            "jac_max_abs_error_mj_jac": mj_jac_err,
            "jac_identity_max_abs_error_z": ident_z,
            "jac_identity_max_abs_error_x": ident_x,
            "n_jac_states": int(sub.shape[0]),
            "n_floor_contacts": n_contacts,
            "n_penetrating_contacts": n_pen,
            "gap_vs_contact_dist_max_abs_error_m": gap_err,
            "gap_min_vs_contact_dist_min_max_abs_error_m": gap_min_err,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
