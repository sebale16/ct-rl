"""Which cheetah capsule endpoints actually touch the floor?

Enumerates the 16 capsule endpoints of dm_control's cheetah (8 collidable
capsules x 2 ends), measures the signed gap ``z_endpoint - radius`` over three
state distributions, and cross-checks against the geom ids MuJoCo itself
reports in ``d.contact``.  The output picks K -- the number of contact slots in
the structured port-Hamiltonian contact port -- from data instead of a guess.

Regimes
  uniform  random qpos over joint ranges, root height in a wide band
  ou       Ornstein-Uhlenbeck exploration rollouts (the off-policy fit regime)
  policy   rollouts from a saved ct-SAC checkpoint (optional)

Heights come from MuJoCo's own forward kinematics (``d.geom_xpos`` /
``d.geom_xmat``); a closed-form planar chain walk derived from mjModel is
verified against it as a self-check, because the production model will use the
closed form rather than calling MuJoCo.

Writes JSON; no plots, no prose.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco  # noqa: E402


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def capsule_endpoints(model):
    """All capsule endpoints as (name, geom_id, body_id, local_pos, radius).

    A capsule's axis is its local z, so the two ends sit at
    ``geom_pos +/- R(geom_quat) @ (0, 0, half_length)`` in the body frame.
    """
    out = []
    for g in range(model.ngeom):
        if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        radius, half = float(model.geom_size[g, 0]), float(model.geom_size[g, 1])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        axis = R.reshape(3, 3) @ np.array([0.0, 0.0, half])
        for sign, tag in ((+1.0, "p"), (-1.0, "m")):
            out.append(
                dict(
                    name=f"{name}_{tag}",
                    geom=g,
                    geom_name=name,
                    body=int(model.geom_bodyid[g]),
                    local=model.geom_pos[g] + sign * axis,
                    radius=radius,
                )
            )
    return out


def planar_chains(model, endpoints):
    """Closed-form planar chain description for every endpoint.

    Every cheetah joint is either a root slide or a hinge about +y, so planar
    rotations compose additively.  For endpoint e the chain is an ordered list
    of (hinge_qpos_adr, dx, dz): the offset accumulated in the frame that
    exists *after* the hinge is applied.
    """
    chains = []
    for e in endpoints:
        segs = []
        b = e["body"]
        tail = np.array([e["local"][0], e["local"][2]])  # (dx, dz) in body frame
        while b != 0:
            hinge = None
            for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                    assert np.allclose(model.jnt_axis[j], [0, 1, 0]), "non-planar hinge"
                    assert np.allclose(model.jnt_pos[j], 0.0), "hinge not at body origin"
                    hinge = int(model.jnt_qposadr[j])
            assert hinge is not None, f"body {b} has no +y hinge"
            segs.append((hinge, float(tail[0]), float(tail[1])))
            bp = model.body_pos[b]
            assert abs(float(bp[1])) < 1e-12, "non-planar body offset"
            b = int(model.body_parentid[b])
            tail = np.array([float(bp[0]), float(bp[2])])
        segs.reverse()  # root-most hinge first
        chains.append(segs)
    return chains


def closed_form_z(model, chains, endpoints, qpos):
    """Endpoint heights from the additive-hinge closed form. qpos (B, nq)."""
    B = qpos.shape[0]
    # world offset of the root body (torso) plus its slide dofs
    root_z0 = float(model.body_pos[1, 2])
    rootz_adr = int(model.jnt_qposadr[1])
    z = np.zeros((B, len(endpoints)))
    for i, segs in enumerate(chains):
        theta = np.zeros(B)
        acc = np.zeros(B)
        for adr, dx, dz in segs:
            theta = theta + qpos[:, adr]
            acc = acc + (-dx * np.sin(theta) + dz * np.cos(theta))
        z[:, i] = root_z0 + qpos[:, rootz_adr] + acc
    return z


def mujoco_endpoint_xz(model, data, endpoints):
    """Endpoint (x, z) from MuJoCo's own kinematics (d.geom_xpos/xmat)."""
    x = np.empty(len(endpoints))
    z = np.empty(len(endpoints))
    for i, e in enumerate(endpoints):
        g = e["geom"]
        R = data.geom_xmat[g].reshape(3, 3)
        half = float(model.geom_size[g, 1])
        sign = 1.0 if e["name"].endswith("_p") else -1.0
        off = sign * (R @ np.array([0.0, 0.0, half]))
        x[i] = data.geom_xpos[g][0] + off[0]
        z[i] = data.geom_xpos[g][2] + off[2]
    return x, z


# --------------------------------------------------------------------------
# state sources
# --------------------------------------------------------------------------

def uniform_states(model, n, rng, rootz_band=(-0.7, 0.7)):
    nq, nv = int(model.nq), int(model.nv)
    q = np.zeros((n, nq))
    q[:, 0] = 0.0  # root x is irrelevant to the gap
    q[:, 1] = rng.uniform(rootz_band[0], rootz_band[1], size=n)
    q[:, 2] = rng.uniform(-np.pi, np.pi, size=n)
    for j in range(3, int(model.njnt)):
        lo, hi = model.jnt_range[j]
        q[:, int(model.jnt_qposadr[j])] = rng.uniform(lo, hi, size=n)
    v = rng.normal(0.0, 1.0, size=(n, nv))
    return q, v


def rollout_states(env, n, policy=None, ou_sigma=0.4, seed=0):
    """OU-exploration (or on-policy) rollout; returns (qpos, qvel) arrays."""
    import torch as th
    from models.noise import OrnsteinUhlenbeckActionNoise

    env.action_space.seed(seed)
    ad = int(np.prod(env.action_space.shape))
    ou = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(ad), sigma=ou_sigma * np.ones(ad), theta=0.15, dt=0.01
    )
    physics = env._env.physics
    Q, V, R = [], [], []
    obs, _ = env.reset()
    for _ in range(n):
        if policy is not None:
            with th.no_grad():
                a_t, _ = policy.act(th.as_tensor(obs, dtype=th.float32).unsqueeze(0))
            a = a_t.squeeze(0).numpy()
        else:
            a = np.clip(ou(), env.action_space.low, env.action_space.high)
        _, _, _, r, no, _, term, trunc, _ = env.step_dt(a)
        Q.append(physics.data.qpos.copy())
        V.append(physics.data.qvel.copy())
        R.append(float(r))
        if term or trunc:
            obs, _ = env.reset()
            ou.reset()
        else:
            obs = no
    return np.asarray(Q), np.asarray(V), float(np.mean(R))


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure(model, data, endpoints, chains, qpos, qvel, want_contacts=True):
    n = qpos.shape[0]
    gaps = np.empty((n, len(endpoints)))
    radii = np.array([e["radius"] for e in endpoints])
    ncon = np.zeros(n, dtype=int)
    pair_counter = Counter()
    geom_counter = Counter()
    floor_states = np.zeros(n, dtype=bool)
    # per contact: (state index, other geom id, contact x, contact z, dist)
    contact_rows = []
    n_distinct_floor_geoms = np.zeros(n, dtype=int)
    # exactness bookkeeping: is MuJoCo's floor-contact set identical to
    # {endpoints with gap < 0}, and does contact.dist equal that gap?
    exact_extra = 0      # MuJoCo contact at an endpoint whose gap >= 0
    exact_missing = 0    # endpoint with gap < 0 and no MuJoCo contact
    dist_vs_gap_max = 0.0
    by_geom = {}
    for i, e in enumerate(endpoints):
        by_geom.setdefault(e["geom"], []).append(i)
    for i in range(n):
        data.qpos[:] = qpos[i]
        data.qvel[:] = qvel[i]
        mujoco.mj_kinematics(model, data)
        ex, ez = mujoco_endpoint_xz(model, data, endpoints)
        gaps[i] = ez - radii
        if want_contacts:
            mujoco.mj_collision(model, data)
            ncon[i] = int(data.ncon)
            seen = set()
            hit_endpoints = set()
            for c in range(int(data.ncon)):
                con = data.contact[c]
                g1, g2 = int(con.geom1), int(con.geom2)
                pair_counter[(g1, g2)] += 1
                geom_counter[g1] += 1
                geom_counter[g2] += 1
                if g1 == 0 or g2 == 0:
                    floor_states[i] = True
                    other = g2 if g1 == 0 else g1
                    # horizontal distance from the contact point to the nearest
                    # endpoint of the same capsule (large => mid-shaft contact)
                    ends = by_geom.get(other, [])
                    dmin = min((abs(float(con.pos[0]) - ex[j]) for j in ends),
                               default=float("nan"))
                    nearest = min(ends, key=lambda j: abs(float(con.pos[0]) - ex[j])) \
                        if ends else -1
                    contact_rows.append(
                        (i, other, float(con.pos[0]), float(con.dist), dmin, nearest)
                    )
                    seen.add(other)
                    if nearest >= 0:
                        hit_endpoints.add(nearest)
                        dist_vs_gap_max = max(
                            dist_vs_gap_max,
                            abs(float(con.dist) - gaps[i, nearest]),
                        )
            n_distinct_floor_geoms[i] = len(seen)
            gap_set = set(np.flatnonzero(gaps[i] < 0.0).tolist())
            exact_extra += len(hit_endpoints - gap_set)
            exact_missing += len(gap_set - hit_endpoints)
    cf = closed_form_z(model, chains, endpoints, qpos) - radii
    cf_err = float(np.abs(cf - gaps).max())
    return dict(
        gaps=gaps,
        ncon=ncon,
        n_distinct_floor_geoms=n_distinct_floor_geoms,
        floor_states=floor_states,
        pair_counter=pair_counter,
        geom_counter=geom_counter,
        contact_rows=contact_rows,
        closed_form_max_abs_err=cf_err,
        exact_extra=exact_extra,
        exact_missing=exact_missing,
        dist_vs_gap_max=dist_vs_gap_max,
    )


def per_point_stats(gaps, endpoints):
    rows = []
    for i, e in enumerate(endpoints):
        g = gaps[:, i]
        rows.append(
            dict(
                name=e["name"],
                geom=e["geom"],
                frac_gap_below_0=float((g < 0.0).mean()),
                frac_gap_below_006=float((g < 0.06).mean()),
                median_gap=float(np.median(g)),
                min_gap=float(g.min()),
                p01_gap=float(np.percentile(g, 1)),
            )
        )
    return rows


def coverage(res, endpoints, subset_names, band=0.06):
    """How many real (floor) contact states a candidate slot set would miss."""
    idx = [i for i, e in enumerate(endpoints) if e["name"] in subset_names]
    assert len(idx) == len(subset_names), "unknown endpoint name in subset"
    gaps = res["gaps"]
    active = (gaps[:, idx] < band).any(axis=1)
    floor = res["floor_states"]
    n_floor = int(floor.sum())
    missed = int((floor & ~active).sum())
    # per-contact-event view: is the contacting geom represented in the subset?
    geoms = {endpoints[i]["geom"] for i in idx}
    rows = res["contact_rows"]
    ev_missed = sum(1 for r in rows if r[1] not in geoms)
    # events whose endpoint is actually in the set, and the same weighted by
    # penetration depth (a proxy for how much contact impulse is being missed)
    ev_nearest_out = sum(1 for r in rows if r[5] not in idx)
    depth_all = sum(abs(r[3]) for r in rows)
    depth_out = sum(abs(r[3]) for r in rows if r[5] not in idx)
    return dict(
        events_whose_nearest_endpoint_is_outside_set=ev_nearest_out,
        frac_events_nearest_endpoint_outside=(ev_nearest_out / len(rows))
        if rows else float("nan"),
        frac_penetration_depth_outside_set=(depth_out / depth_all)
        if depth_all > 0 else float("nan"),
        n_contact_states=n_floor,
        frac_states_with_contact=float(floor.mean()),
        missed_contact_states=missed,
        frac_contact_states_missed=(missed / n_floor) if n_floor else float("nan"),
        n_contact_events=len(rows),
        events_on_geoms_outside_set=ev_missed,
        frac_events_outside_set=(ev_missed / len(rows)) if rows else float("nan"),
    )


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="label=/path/to/ckpt.pth; repeatable")
    ap.add_argument("--mode", default="mbq_structured_quad_cforce_buf1m")
    ap.add_argument("--env_id", default="cheetah-run")
    ap.add_argument("--out", default="results/cheetah_contact_endpoint_census.json")
    ap.add_argument("--dump_npz", default=None,
                    help="also dump raw per-state endpoint gaps here")
    args = ap.parse_args()

    from environment.dmc import DMCContinuousEnv

    rng = np.random.default_rng(args.seed)
    env = DMCContinuousEnv(
        "cheetah", "run", time_sampling="uniform", dt=0.01, physics_dt=0.002,
        episode_duration=20.0, seed=args.seed,
    )
    physics = env._env.physics
    model = physics.model.ptr if hasattr(physics.model, "ptr") else physics.model
    data = mujoco.MjData(model)

    endpoints = capsule_endpoints(model)
    chains = planar_chains(model, endpoints)
    print(f"{len(endpoints)} capsule endpoints")

    # mj_kinematics + mj_collision must agree with a full mj_forward on the
    # contact set; everything below uses the cheap path.
    qc, vc = uniform_states(model, 300, np.random.default_rng(12345))
    mismatch = 0
    for i in range(qc.shape[0]):
        data.qpos[:] = qc[i]; data.qvel[:] = vc[i]
        mujoco.mj_kinematics(model, data); mujoco.mj_collision(model, data)
        a = sorted((int(data.contact[c].geom1), int(data.contact[c].geom2))
                   for c in range(int(data.ncon)))
        data.qpos[:] = qc[i]; data.qvel[:] = vc[i]
        mujoco.mj_forward(model, data)
        b = sorted((int(data.contact[c].geom1), int(data.contact[c].geom2))
                   for c in range(int(data.ncon)))
        mismatch += int(a != b)
    print(f"mj_collision vs mj_forward contact-set mismatches: {mismatch}/300")

    regimes = {}
    raw = {}

    q, v = uniform_states(model, args.n, rng)
    raw["uniform"] = measure(model, data, endpoints, chains, q, v)

    q, v, r_ou = rollout_states(env, args.n, policy=None, seed=args.seed)
    raw["ou"] = measure(model, data, endpoints, chains, q, v)
    mean_reward = {"ou": r_ou}

    for spec in args.checkpoint:
        from common.utils import load_ct_hyperparams_from_table
        from models.actor_q_critic import ActorQCriticModel

        label, _, path = spec.partition("=")
        _, _, model_kwargs, _, _ = load_ct_hyperparams_from_table(
            "ct_sac", args.env_id, args.mode
        )
        pol = ActorQCriticModel(
            observation_space=env.observation_space,
            action_space=env.action_space, device="cpu", **model_kwargs,
        )
        pol.load_state(path)
        q, v, r_pi = rollout_states(env, args.n, policy=pol, seed=args.seed + 1)
        raw[label] = measure(model, data, endpoints, chains, q, v)
        mean_reward[label] = r_pi
        # root pitch tells us whether the cheetah is upright or has flipped
        pitch = np.mod(q[:, 2] + np.pi, 2 * np.pi) - np.pi
        raw[label]["frac_inverted"] = float((np.abs(pitch) > np.pi / 2).mean())
        print(f"{label}: mean per-step reward {r_pi:.4f}, "
              f"|pitch|>90deg {raw[label]['frac_inverted']:.4f} (OU {r_ou:.4f})")

    # candidate slot sets, smallest first
    feet = ["bfoot_m", "bfoot_p", "ffoot_m", "ffoot_p"]
    shins = ["bshin_p", "bshin_m", "fshin_p", "fshin_m"]
    cand = {
        "K2_toes": ["bfoot_m", "ffoot_m"],
        "K3_toes_fshin_m": ["bfoot_m", "ffoot_m", "fshin_m"],
        "K4_feet": feet,
        "K4_toes_bshinp_fshinm": ["bfoot_m", "ffoot_m", "bshin_p", "fshin_m"],
        "K6_feet_bshinp_fshinm": feet + ["bshin_p", "fshin_m"],
        "K6_feet_headp_torsop": feet + ["head_p", "torso_p"],
        "K8_feet_shins": feet + shins,
        "K8_feet_bshinp_fshinm_headp_torsop": feet
        + ["bshin_p", "fshin_m", "head_p", "torso_p"],
        "K10_feet_shins_headp_torsop": feet + shins + ["head_p", "torso_p"],
        "K12_legs_only": feet + shins
        + ["bthigh_p", "bthigh_m", "fthigh_p", "fthigh_m"],
        "K16_all": [e["name"] for e in endpoints],
    }

    out = dict(
        n_per_regime=args.n,
        mj_collision_vs_mj_forward_mismatches=mismatch,
        checkpoint=args.checkpoint,
        mean_step_reward=mean_reward,
        endpoints=[
            dict(name=e["name"], geom=e["geom"], geom_name=e["geom_name"],
                 body=int(e["body"]), radius=e["radius"],
                 local_pos=[float(x) for x in e["local"]])
            for e in endpoints
        ],
        regimes={},
    )
    for k, res in raw.items():
        gname = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
        out["regimes"][k] = dict(
            closed_form_vs_mujoco_max_abs_err=res["closed_form_max_abs_err"],
            endpoint_gap_set_extra_vs_mujoco=res["exact_extra"],
            endpoint_gap_set_missing_vs_mujoco=res["exact_missing"],
            mujoco_dist_vs_endpoint_gap_max_abs_err=res["dist_vs_gap_max"],
            frac_states_with_any_contact=float((res["ncon"] > 0).mean()),
            frac_states_with_floor_contact=float(res["floor_states"].mean()),
            mean_ncon=float(res["ncon"].mean()),
            max_ncon=int(res["ncon"].max()),
            ncon_histogram={
                str(int(v)): int(c)
                for v, c in zip(*np.unique(res["ncon"], return_counts=True))
            },
            distinct_floor_geoms_histogram={
                str(int(v)): int(c)
                for v, c in zip(*np.unique(res["n_distinct_floor_geoms"],
                                           return_counts=True))
            },
            contact_geom_counts={
                f"{g}:{gname(g)}": c for g, c in sorted(res["geom_counter"].items())
            },
            contact_pair_counts={
                f"{gname(a)}|{gname(b)}": c
                for (a, b), c in sorted(res["pair_counter"].items(), key=lambda kv: -kv[1])
            },
            frac_inverted=res.get("frac_inverted"),
            per_point=per_point_stats(res["gaps"], endpoints),
            coverage={name: coverage(res, endpoints, s) for name, s in cand.items()},
        )
        # how far MuJoCo contact points sit from the nearest endpoint of the
        # same capsule (mid-shaft contacts are not representable by endpoints)
        rows = res["contact_rows"]
        out["regimes"][k]["n_floor_contact_events"] = len(rows)
        if rows:
            d = np.array([r[4] for r in rows])
            nearest = Counter(endpoints[r[5]]["name"] for r in rows if r[5] >= 0)
            out["regimes"][k]["contact_to_nearest_endpoint_x_dist"] = dict(
                median=float(np.median(d)),
                p90=float(np.percentile(d, 90)),
                p99=float(np.percentile(d, 99)),
                max=float(d.max()),
                frac_beyond_5cm=float((d > 0.05).mean()),
            )
            out["regimes"][k]["nearest_endpoint_event_counts"] = dict(
                sorted(nearest.items(), key=lambda kv: -kv[1])
            )

    if args.dump_npz:
        np.savez_compressed(
            args.dump_npz,
            names=np.array([e["name"] for e in endpoints]),
            **{f"gaps_{k}": v["gaps"].astype(np.float32) for k, v in raw.items()},
            **{f"floor_{k}": v["floor_states"] for k, v in raw.items()},
            **{f"events_{k}": np.array([(r[0], r[1], r[3], r[5]) for r in v["contact_rows"]],
                                       dtype=np.float64) for k, v in raw.items()},
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")

    for k, r in out["regimes"].items():
        print(f"\n=== {k}  closed-form err {r['closed_form_vs_mujoco_max_abs_err']:.2e} "
              f"floor-contact states {r['frac_states_with_floor_contact']:.4f} ===")
        print("  geom contact counts:", r["contact_geom_counts"])
        for row in r["per_point"]:
            print(f"  {row['name']:>10s} <0 {row['frac_gap_below_0']:.4f}  "
                  f"<0.06 {row['frac_gap_below_006']:.4f}  "
                  f"med {row['median_gap']:+.3f}  min {row['min_gap']:+.3f}")
        for name, c in r["coverage"].items():
            print(f"  {name:>36s}: miss {c['frac_contact_states_missed']:.4f} of "
                  f"{c['n_contact_states']} contact states; events off-set "
                  f"{c['frac_events_nearest_endpoint_outside']:.4f}; depth off-set "
                  f"{c['frac_penetration_depth_outside_set']:.4f}")


if __name__ == "__main__":
    main()
