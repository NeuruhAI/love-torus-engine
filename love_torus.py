import argparse
import json
import math
from typing import Dict, List, Tuple

# Locked system constants
ALPHA, BETA, GAMMA = 0.25, 0.20, 0.15
DELTA_C, K, V_MIN = 0.50, 10.0, 0.10
OMEGA = math.pi / 2.0
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def step_love_torus(
    state: Dict[str, float],
    A: float,
    E: float,
    G: float,
    F: float,
    Delta: float,
    dt: float = 1.0,
) -> Dict[str, float]:
    """Single deterministic transition on the expanding Horn Torus."""
    aperture_now = 1.0 / (1.0 + math.exp(-K * (Delta - DELTA_C)))
    P = max(state["P"], aperture_now)

    q = A * E * P
    L = state["L"] + q * dt

    dV = ALPHA * (A * E) - BETA * G + GAMMA * F
    V = max(V_MIN, state["V"] + dV * dt)

    phi = (OMEGA * L) % (2.0 * math.pi)
    u = (state["u"] + GOLDEN_ANGLE) % (2.0 * math.pi)
    r = V * (1.0 - math.cos(phi))
    x = r * math.cos(u)
    y = r * math.sin(u)
    z = V * math.sin(phi)

    return {
        "P": P,
        "q": q,
        "L": L,
        "V": V,
        "phi": phi,
        "u": u,
        "r": r,
        "x": x,
        "y": y,
        "z": z,
    }


def interpolate_trajectory(
    keyframes: List[Tuple[float, float, float, float, float]],
    substeps: int,
) -> List[Tuple[float, float, float, float, float]]:
    """Linearly expands coarse keyframes for dense particle export."""
    if substeps <= 1:
        return keyframes
    dense = []
    for i in range(len(keyframes) - 1):
        start, end = keyframes[i], keyframes[i + 1]
        for s in range(substeps):
            w = s / substeps
            dense.append(tuple(start[j] + (end[j] - start[j]) * w for j in range(5)))
    dense.append(keyframes[-1])
    return dense


def main() -> None:
    parser = argparse.ArgumentParser(description="Horn Torus dynamical engine")
    parser.add_argument("--substeps", type=int, default=1, help="Substeps per phase (use 200+ for WebGL export)")
    parser.add_argument("--export", type=str, default="", help="Optional path to write JSON trajectory")
    args = parser.parse_args()

    keyframes = [
        (0.80, 0.10, 0.70, 0.20, 0.20),  # 1. Guarded
        (0.60, 0.30, 0.40, 0.30, 0.45),  # 2. Building Pressure
        (0.70, 0.70, 0.20, 0.50, 0.80),  # 3. The Crack
        (0.80, 0.80, 0.30, 0.60, 0.40),  # 4. Sloppy In-Between
        (0.90, 0.90, 0.10, 0.80, 0.20),  # 5. Outward Bloom
        (0.90, 0.95, 0.05, 0.90, 0.15),  # 6. Return Cascade
        (0.85, 0.90, 0.10, 0.85, 0.10),  # 7. Re-Entry (Saved)
    ]

    dt = 1.0 / max(1, args.substeps)
    inputs = interpolate_trajectory(keyframes, args.substeps)
    state = {"P": 0.0, "L": 0.0, "V": 1.0, "u": 0.0}
    records = []

    for idx, inp in enumerate(inputs, start=1):
        state = step_love_torus(state, *inp, dt=dt)
        records.append({"step": idx, **{k: round(v, 6) for k, v in state.items()}})
        if args.substeps == 1:
            print(
                f"t={idx} | P={state['P']:.3f} q={state['q']:.3f} L={state['L']:.3f} "
                f"V={state['V']:.3f} | phi={math.degrees(state['phi']):5.1f}deg "
                f"r={state['r']:.3f} z={state['z']:+.3f}"
            )

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump({"dt": dt, "count": len(records), "frames": records}, f)
        print(f"Exported {len(records)} frames to {args.export}")


if __name__ == "__main__":
    main()
