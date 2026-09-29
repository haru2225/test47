#!/usr/bin/env python3
"""Corrected generation for a test47-trained checkpoint.

WHY THIS FILE EXISTS: test47.py's `generate()` (and DM2's own demo generation code it is ported
from) adds noise of magnitude `sigma` itself at every annealing step, fresh and independent each
time. Summed over the full 2900-step schedule (sigma: 1.0 -> 0.001), the injected noise ALONE has
variance sum(sigma_i**2) =~ 968, i.e. a standard deviation of ~31 Angstrom -- more than TWICE the
13.57 A unit cell -- regardless of whether the model's prediction is any good. Verified empirically:
a real test47 run (18-reference training, l<=5, loss converged to ~0.015) produced a "generated"
structure with a mean nearest-neighbor distance of 1.24 A and a minimum of 0.24 A (atoms on top of
each other), and completely flat bond/angle histograms -- the generation step destroys structure
by construction, independent of training quality.

THE FIX: use the properly SDE-consistent per-step noise scale sqrt(dv), dv = sigma_i**2 -
sigma_{i+1}**2 (the *change* in noise level, not the noise level itself), matching the VE-SDE
reverse step already used and verified in this project's test42/43/44 (toy-model/SiO2-CG/test42.py
etc: `pos += dv/sigma * pred + sqrt(dv) * randn`, with `pred` there being the network's own
"sigma*score" output). test47's network is trained differently -- to predict `dx`, the raw
displacement added by RattleParticles (dx = sigma*eps), i.e. an eps/x0-prediction denoiser, not a
sigma-normalized score. Via Tweedie's formula, eps ~= -sigma*score, so sigma*score ~= -pred_dx/sigma,
giving the corrected step used below:

    dv = sigma_i**2 - sigma_{i+1}**2
    pos <- pos - (dv / sigma_i**2) * pred_dx + sqrt(dv) * randn        (annealing phase)
    pos <- pos - pred_dx                                               (final zero-noise polish)

This is a NEW script, not a modification of test47.py's own `generate()` (which is left as the
def-for-def-adjacent DM2 port it was), so both behaviours stay directly comparable.

Also adds a same-species (Si-Si, O-O) minimum-distance check to the saved metrics, which
test47.py's own metrics.json lacks -- the collapse failure mode above is exactly what that check
is for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from torch_geometric.data import Data

import test47 as t47

SCRIPT_DIR = Path(__file__).resolve().parent


def sigma_schedule(sigma_max: float, sigma_min: float, steps: int) -> np.ndarray:
    """Geometric (log-spaced), matching test42-44's VE-SDE schedule -- NOT DM2's linear one."""
    return np.geomspace(sigma_max, sigma_min, steps + 1)


def same_species_min_distance(positions: np.ndarray, numbers: np.ndarray, cell: np.ndarray,
                               number: int) -> Optional[float]:
    indices = np.where(numbers == number)[0]
    if len(indices) < 2:
        return None
    i, j, d = primitive_neighbor_list("ijd", pbc=[True, True, True], cell=cell, positions=positions[indices],
                                       cutoff=min(cell.diagonal()) / 2 - 1e-6)
    return float(d.min()) if len(d) else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="test47.py state-dict checkpoint.")
    p.add_argument("--crystal-data", type=Path, default=t47.DEFAULT_CRYSTAL_DATA)
    p.add_argument("--replicate", type=int, nargs=3, default=(1, 1, 1), metavar=("NX", "NY", "NZ"))
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--irreps-hidden", default="64x0e + 32x1e + 16x2e + 8x3e + 4x4e + 2x5e",
                   help="Must match the checkpoint's architecture.")
    p.add_argument("--irreps-edge", default="4x0e + 4x1e + 2x2e + 2x3e + 1x4e + 1x5e",
                   help="Must match the checkpoint's architecture.")
    p.add_argument("--num-convs", type=int, default=3)
    p.add_argument("--init", choices=("crystal", "random"), default="crystal")
    p.add_argument("--steps", type=int, default=2900, help="Annealing steps, sigma-max -> sigma-min.")
    p.add_argument("--polish-steps", type=int, default=100, help="Extra zero-noise steps at the end.")
    p.add_argument("--sigma-max", type=float, default=0.75,
                   help="Training's own sigma_max (RattleParticles default); do not exceed what "
                        "the model was actually trained on, unlike DM2's own gen-time 1.0.")
    p.add_argument("--sigma-min", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "generate-v2-output")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Submit to a GPU node or use --allow-cpu for a smoke test.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    unit_atoms = ase.io.read(args.crystal_data.expanduser().resolve(), format="lammps-data")
    atoms = unit_atoms.repeat(tuple(args.replicate))
    numbers = np.asarray(atoms.numbers)
    si_mask = numbers == 14

    model = t47.build_model(args.cutoff, device, args.irreps_hidden, args.irreps_edge, args.num_convs)
    model.load_state_dict(t47.torch_load(args.checkpoint, map_location="cpu"))
    model.eval()

    start_positions = t47.initial_positions(atoms, args.init)
    pos = torch.tensor(start_positions, dtype=torch.float32, device=device)
    species = torch.tensor((numbers == 14).astype(np.int64), device=device)
    cell = torch.tensor(np.asarray(atoms.cell), dtype=torch.float32, device=device)
    box = np.diag(np.asarray(atoms.cell))

    sigmas = sigma_schedule(args.sigma_max, args.sigma_min, args.steps)
    print(f"init={args.init}  steps={args.steps}+{args.polish_steps}  "
          f"sigma {args.sigma_max} -> {args.sigma_min} (geometric)", flush=True)

    with torch.no_grad():
        for step in range(args.steps):
            sigma_i, sigma_next = float(sigmas[step]), float(sigmas[step + 1])
            dv = sigma_i**2 - sigma_next**2
            data = t47.graph_on_device(
                Data(x=species, pos=pos, cell=cell, pbc=atoms.pbc, numbers=numbers),
                args.cutoff, numbers,
            )
            pred_dx = model(data)
            pos = pos - (dv / sigma_i**2) * pred_dx + np.sqrt(dv) * torch.randn_like(pos)
            pos = torch.remainder(pos, pos.new_tensor(box))
            if (step + 1) % 100 == 0 or step == args.steps - 1:
                print(f"anneal step {step + 1}/{args.steps}  sigma={sigma_i:.4g}", flush=True)

        for step in range(args.polish_steps):
            data = t47.graph_on_device(
                Data(x=species, pos=pos, cell=cell, pbc=atoms.pbc, numbers=numbers),
                args.cutoff, numbers,
            )
            pos = pos - model(data)
            pos = torch.remainder(pos, pos.new_tensor(box))
        print(f"polish steps: {args.polish_steps}", flush=True)

    final_positions = pos.detach().cpu().numpy()
    final_wrapped = t47.wrap_positions(final_positions, np.asarray(atoms.cell))
    final_atoms = Atoms(numbers=numbers, positions=final_wrapped, cell=atoms.cell, pbc=True)
    ase.io.write(output_dir / "final_structure.extxyz", final_atoms)

    reference_positions = t47.wrap_positions(np.asarray(atoms.positions), np.asarray(atoms.cell))
    generated_bonds, generated_angles = t47.bond_and_angle_stats(final_wrapped, numbers, np.asarray(atoms.cell))
    reference_bonds, reference_angles = t47.bond_and_angle_stats(reference_positions, numbers, np.asarray(atoms.cell))
    t47.save_histogram(reference_bonds, generated_bonds, output_dir / "bond_comparison.png",
                        "Si-O distance (Angstrom)", "test47 generate-v2: Si-O bond length", 1.61)
    t47.save_histogram(reference_angles, generated_angles, output_dir / "angle_comparison.png",
                        "O-Si-O angle (degree)", "test47 generate-v2: O-Si-O angle", 109.47)

    si_si_min = same_species_min_distance(final_wrapped, numbers, np.asarray(atoms.cell), 14)
    o_o_min = same_species_min_distance(final_wrapped, numbers, np.asarray(atoms.cell), 8)
    metrics = {
        "init": args.init,
        "steps": args.steps,
        "polish_steps": args.polish_steps,
        "sigma_max": args.sigma_max,
        "sigma_min": args.sigma_min,
        "reference_bond_mean_angstrom": float(reference_bonds.mean()),
        "reference_bond_std_angstrom": float(reference_bonds.std()),
        "generated_bond_mean_angstrom": float(generated_bonds.mean()),
        "generated_bond_std_angstrom": float(generated_bonds.std()),
        "reference_angle_mean_degree": float(reference_angles.mean()),
        "reference_angle_std_degree": float(reference_angles.std()),
        "generated_angle_mean_degree": float(generated_angles.mean()),
        "generated_angle_std_degree": float(generated_angles.std()),
        "generated_si_si_min_distance_angstrom": si_si_min,
        "generated_o_o_min_distance_angstrom": o_o_min,
        "checkpoint_sha256": t47.file_sha256(args.checkpoint),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in (
        "generated_bond_mean_angstrom", "generated_angle_mean_degree",
        "generated_si_si_min_distance_angstrom", "generated_o_o_min_distance_angstrom")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
