#!/usr/bin/env python3
"""test47 = test45 (l<=3) with the NequIP irreps raised further to l<=5 by default (hidden
64x0e+32x1e+16x2e+8x3e+4x4e+2x5e, edge 4x0e+4x1e+2x2e+2x3e+1x4e+1x5e), and --replicate defaults
to 1 1 1 (192-atom unit cell, not the 1536-atom 2x2x2 supercell) instead of test45's 2 2 2 --
requested as-is, so the box (13.6 A, half-box 6.79 A) is again SMALLER than the default
--large-cutoff (10.0 A): see fix #1 below, which this reintroduces (pass --large-cutoff <= 6.5,
e.g. together with a correspondingly small --cutoff, if you want that fixed instead). The
irreps/num_convs are CLI-configurable (--irreps-hidden/--irreps-edge/--num-convs) so a CUDA OOM
from the l<=5 tensor products can be narrowed down without editing the file.

test47 adds two further changes on top of test46:

3. TRAINING REFERENCE DIVERSITY: test33/test45/test46 train on a SINGLE ideal crystal structure
   duplicated --duplicate times, so the only diversity the model ever sees is the fresh
   RattleParticles noise drawn each step -- the reference geometry itself never changes. test47
   instead defaults to --reference-frames data/reference_frames.npz, 18 real thermal snapshots
   (9 each from md/traj_0.lammpstrj and md/traj_1.lammpstrj's beta-cristobalite NPT MD, index
   2000::1000) covering genuine thermal displacement away from the ideal lattice, not just
   synthetic noise. Each of the N frames is duplicated --duplicate times (dataset size becomes
   N x --duplicate). Pass --single-reference to fall back to test46's single-structure behaviour.
4. RANDOM-START GENERATION: test33/test45/test46 always start generation from the ideal crystal
   itself (crystal_atoms), which only tests whether the model can denoise back to a known-good
   structure, not whether it can generate a crystal from nothing. --init random (new; default
   remains "crystal" for backward compatibility) instead starts from atoms placed uniformly at
   random in the cell. Run both --init crystal and --init random (with different --output-dir) to
   compare whether generation actually works from scratch.

Both training and generation now also append every progress message to <output-dir>/progress.log
(see the log() helper) as they happen, in addition to stdout -- so progress is visible as a file
inside the output directory itself while the job is still running, not only via the PBS stdout
log (whose filename/location depends on the job ID).

Train test33's DM2 NequIP denoiser on a replicated SiO2 crystal, generate
a structure, and export a Si-only CG trajectory.

This is the supercomputer-portable version of ``toy-model/test33.py``.
Unlike ``test32-CGMD.py`` (which only regenerates from an already-trained
checkpoint), this script also TRAINS the model from scratch when no final
checkpoint is present, and is resumable across PBS walltime limits for BOTH
the training phase and the generation phase independently.

Two things fixed here relative to the very first (locally-run) test33.py
attempt, which generated badly wrong crystal structures -- narrow bond/angle
histograms collapsed onto ~4 discrete, non-tetrahedral values instead of a
continuous spread around the reference's 108.82 degree mean:

1. The reference cell is replicated 2x2x2 (--replicate, default 2 2 2):
   the raw 192-atom unit cell is only 13.573 Angstrom across, so half the
   box (6.79 A) is SMALLER than DM2's own large_cutoff=10.0 A (tuned for
   their 3000-atom, ~36 A glass box). Verified with ASE's
   primitive_neighbor_list that at cutoff=10.0, 69.5% of atom pairs are
   connected through 2-4 DISTINCT periodic images simultaneously (a genuine
   minimum-image-convention violation; 0% at cutoff<=6.5). Replicating to
   ~27.1 A (half-box 13.6 A) removes this with margin to spare, and also
   keeps the network's receptive field (num_convs=3 * cutoff=5.0 = ~15 A)
   from spanning essentially the whole periodic cell.
2. Generation now builds its graph at --cutoff (5.0 A, matching the trained
   message-passing radius) instead of a separate large-cutoff value. Training
   only ever message-passes over edges <= cutoff (DownselectEdges is applied
   after every rattle); using a wider graph at generation time -- as the
   original toy-model/test33.py did, copying DM2's own demo code, which
   builds its LARGE (pre-rattle) graph directly with no downselection at
   generation time -- feeds the model a systematically denser graph than
   anything it saw in training. This mismatch is orthogonal to fix #1 (it
   would exist even in an arbitrarily large box) and matches the convention
   already used by the sibling script test32-CGMD.py.

All paths are command-line arguments or are resolved relative to the cloned
DM2 repository. Both the training and the generation restart data are saved
periodically, so submitting the same PBS file again continues an interrupted
job from wherever it left off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

import ase.io
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from ase.neighborlist import primitive_neighbor_list
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP
from graphite.transforms import DownselectEdges, RattleParticles


warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._check")

SCRIPT_DIR = Path(__file__).resolve().parent
DM2_ROOT = Path(os.environ.get("DM2_ROOT", SCRIPT_DIR.parent / "DM2")).resolve()  # only printed; graphite is found via PYTHONPATH
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "test47_sio2_crystal_2x2x2_nequip.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "test47-output"
DEFAULT_CRYSTAL_DATA = SCRIPT_DIR / "data" / "silica_beta_cristobalite_init.data"
DEFAULT_REFERENCE_FRAMES = SCRIPT_DIR / "data" / "reference_frames.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--crystal-data", type=Path, default=DEFAULT_CRYSTAL_DATA,
        help="LAMMPS-data-format SiO2 crystal unit cell (default: bundled data/silica_beta_cristobalite_init.data).",
    )
    parser.add_argument("--replicate", type=int, nargs=3, default=(1, 1, 1), metavar=("NX", "NY", "NZ"))
    parser.add_argument(
        "--reference-frames", type=Path, default=DEFAULT_REFERENCE_FRAMES,
        help="test47 addition: an .npz of MULTIPLE thermal MD snapshots (positions, cell_lengths, numbers; "
             "see data/reference_frames_metadata.json) used as TRAINING references instead of test45/46's "
             "single ideal crystal duplicated --duplicate times. Default: bundled "
             "data/reference_frames.npz (18 beta-cristobalite NPT snapshots from md/traj_0.lammpstrj and "
             "md/traj_1.lammpstrj). Each of the N frames is duplicated --duplicate times, so the training "
             "dataset size becomes N x --duplicate (was just --duplicate). Ignored if it does not exist or "
             "--single-reference is passed.")
    parser.add_argument(
        "--single-reference", action="store_true",
        help="Ignore --reference-frames and train on --crystal-data's single ideal structure duplicated "
             "--duplicate times, matching test45/test46's behaviour.")
    parser.add_argument(
        "--init", choices=("crystal", "random"), default="crystal",
        help="Starting configuration for GENERATION (test47 addition). 'crystal' (default, matches "
             "test33/test45/test46): start from the ideal --crystal-data structure with fresh noise added "
             "each step -- this only tests whether the model can denoise BACK to a known-good structure, "
             "not whether it can generate a crystal from nothing. 'random': start from atoms placed "
             "uniformly at random in the cell -- a real test of crystal generation. Run both (with "
             "different --output-dir) to compare.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                         help="Final trained state-dict path. If it already exists, training is skipped.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--irreps-hidden", default="64x0e + 32x1e + 16x2e + 8x3e + 4x4e + 2x5e",
                         help="NequIP hidden-layer irreps (test47 default: l<=5; test33 used '64x0e + 32x1e', l<=1; test45 used l<=3). "
                              "Lower multiplicities on the l=2..5 terms (e.g. '64x0e + 32x1e + 8x2e + 4x3e + 2x4e + 1x5e') "
                              "to cut e3nn tensor-product memory if you hit CUDA OOM.")
    parser.add_argument("--irreps-edge", default="4x0e + 4x1e + 2x2e + 2x3e + 1x4e + 1x5e",
                         help="Spherical-harmonics irreps used for edges (test47 default: l<=5; test33 used "
                              "'4x0e + 4x1e + 2x2e', l<=2; test45 used l<=3). Must be consistent with --irreps-hidden.")
    parser.add_argument("--num-convs", type=int, default=3, help="Number of NequIP interaction layers.")
    parser.add_argument("--large-cutoff", type=float, default=5.0,
                         help="Pre-rattle graph margin used only during TRAINING dataset construction. "
                              "test47 default (5.0, equal to --cutoff) avoids the duplicate-periodic-image "
                              "bug (see module docstring) at the default --replicate 1 1 1 (half-box 6.79 A); "
                              "test33/test45 used 10.0, tuned for their larger boxes. Note this leaves no "
                              "rattle margin above --cutoff: a pair just beyond --large-cutoff before noise "
                              "can never appear as an edge even if noise brings it within --cutoff.")
    parser.add_argument("--learn-rate", type=float, default=2e-4)
    parser.add_argument("--num-updates", type=int, default=30_000)
    parser.add_argument("--duplicate", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--sigma-max", type=float, default=0.75, help="RattleParticles training sigma_max.")
    parser.add_argument("--training-checkpoint-epochs", type=int, default=5,
                         help="Save a training restart file every N epochs.")
    parser.add_argument("--gen-noisy-steps", type=int, default=2_900)
    parser.add_argument("--gen-polish-steps", type=int, default=100)
    parser.add_argument("--gen-max-sigma", type=float, default=1.0)
    parser.add_argument("--generation-checkpoint-steps", type=int, default=100)
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=11.5,
        help="Save restart data and exit before PBS walltime (0 disables the guard).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-cpu", action="store_true", help="Only for a small local smoke test.")
    parser.add_argument("--force", action="store_true", help="Overwrite a completed generation run.")
    args = parser.parse_args()

    if args.cutoff <= 0 or args.large_cutoff <= 0:
        parser.error("--cutoff and --large-cutoff must be positive")
    if args.large_cutoff < args.cutoff:
        parser.error("--large-cutoff must be >= --cutoff")
    if args.gen_noisy_steps < 0 or args.gen_polish_steps < 0:
        parser.error("Generation step counts cannot be negative")
    if args.generation_checkpoint_steps < 1 or args.training_checkpoint_epochs < 1:
        parser.error("Checkpoint intervals must be positive")
    return args


# =============================================================================
# Generic helpers -- shared with test32-CGMD.py's conventions.
# =============================================================================


def torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def log(output_dir: Path, message: str) -> None:
    """Print AND append to output_dir/progress.log -- test47 addition, so the
    running job's progress is visible as a file inside the output directory
    itself while training/generation is still in progress, not only via the
    PBS stdout log (whose filename depends on the job ID)."""
    print(message, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as stream:
        stream.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def capture_rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def deadline_reached(deadline: Optional[float]) -> bool:
    # Leave two minutes for writing restart data and clean Singularity exit.
    return deadline is not None and time.monotonic() >= deadline - 120.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_positions(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = positions @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    return fractional @ cell


def bond_and_angle_stats(positions, numbers, cell) -> tuple[np.ndarray, np.ndarray]:
    si_indices = np.where(numbers == 14)[0]
    o_indices = np.where(numbers == 8)[0]
    displacement = positions[si_indices, None, :] - positions[None, o_indices, :]
    fractional = displacement @ np.linalg.inv(cell)
    fractional -= np.round(fractional)
    displacement = fractional @ cell
    distances = np.linalg.norm(displacement, axis=-1)
    bonds, angles = [], []
    for si_local in range(len(si_indices)):
        nearest = np.argsort(distances[si_local])[:4]
        bonds.extend(distances[si_local, nearest].tolist())
        vectors = displacement[si_local, nearest]
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
        for first in range(4):
            for second in range(first + 1, 4):
                cosine = np.clip(np.dot(vectors[first], vectors[second]), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosine)))
    return np.asarray(bonds), np.asarray(angles)


def save_histogram(reference, generated, target, xlabel, title, marker) -> None:
    figure, axis = plt.subplots()
    axis.hist(reference, bins=60, density=True, histtype="step", label="reference", linewidth=2)
    axis.hist(generated, bins=60, density=True, histtype="step", label="generated", linewidth=2)
    axis.axvline(marker, color="gray", linestyle=":")
    axis.set(xlabel=xlabel, title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(target, dpi=160)
    plt.close(figure)


# =============================================================================
# Model -- identical architecture to test32-CGMD.py so both share the same
# NequIP hyperparameters (only the checkpoint's learned weights differ).
# =============================================================================


class InitialEmbedding(nn.Module):
    def __init__(self, num_species: int, cutoff: float):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data: Data) -> Data:
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def build_model(cutoff: float, device: torch.device, irreps_hidden: str, irreps_edge: str,
                 num_convs: int) -> NequIP:
    return NequIP(
        init_embed=InitialEmbedding(num_species=2, cutoff=cutoff),
        irreps_node_x="8x0e",
        irreps_node_z="8x0e",
        irreps_hidden=irreps_hidden,  # test47 default: l up to 5 (test33: up to l=1, test45: up to l=3)
        irreps_edge=irreps_edge,  # test47 default: l up to 5 (test33: up to l=2, test45: up to l=3)
        irreps_out="1x1e",
        num_convs=num_convs,
        radial_neurons=[16, 64],
        num_neighbors=12,
    ).to(device)


def graph_on_device(data: Data, cutoff: float, numbers: np.ndarray) -> Data:
    """Build ASE's periodic graph on CPU and transfer only its edges to the GPU."""
    device = data.pos.device
    positions = data.pos.detach().cpu().numpy()
    cell = data.cell.detach().cpu().numpy()
    pbc = data.pbc if isinstance(data.pbc, (list, tuple, np.ndarray)) else data.pbc.detach().cpu().numpy()
    i, j, displacement = primitive_neighbor_list(
        "ijD", cutoff=cutoff, pbc=pbc, cell=cell, positions=positions, numbers=numbers,
    )
    data.edge_index = torch.from_numpy(np.stack((i, j))).long().to(device)
    data.edge_attr = torch.from_numpy(displacement).float().to(device)
    return data


# =============================================================================
# Training dataset -- DM2's own trick (duplicate one reference structure,
# learn to reverse fresh noise every step), same as toy-model/test33.py.
# =============================================================================


def ase_graph_cpu(data: Data, cutoff: float) -> Data:
    i, j, d = primitive_neighbor_list(
        "ijD", cutoff=cutoff, pbc=data.pbc, cell=data.cell, positions=data.pos.numpy(), numbers=data.numbers,
    )
    data.edge_index = torch.tensor(np.stack((i, j)), dtype=torch.long)
    data.edge_attr = torch.tensor(d, dtype=torch.float)
    return data


class PeriodicStructureDataset(Dataset):
    """test47: accepts a LIST of reference structures (e.g. several thermal MD snapshots), each
    duplicated `duplicate` times -- was a single structure in test33/test45/test46, so the only
    training diversity was the fresh RattleParticles noise draw, never the reference geometry."""

    def __init__(self, atoms_list, large_cutoff: float, duplicate: int):
        super().__init__(None, transform=None, pre_transform=None)
        self.dataset = []
        for atoms in atoms_list:
            species = LabelEncoder().fit_transform(atoms.numbers)
            base = Data(
                x=torch.tensor(species).long(),
                pos=torch.tensor(atoms.positions).float(),
                cell=atoms.cell,
                pbc=atoms.pbc,
                numbers=atoms.numbers,
            )
            base = ase_graph_cpu(base, large_cutoff)
            self.dataset.extend(base.clone() for _ in range(duplicate))

    def len(self) -> int:
        return len(self.dataset)

    def get(self, idx: int):
        return self.dataset[idx]


def load_reference_frames(path: Path, replicate: tuple[int, int, int]) -> list:
    """test47: load the bundled multi-snapshot thermal reference (see data/reference_frames_metadata.json)
    and replicate each frame the same way --replicate replicates the single ideal --crystal-data structure."""
    archive = np.load(path)
    positions, cell_lengths, numbers = archive["positions"], archive["cell_lengths"], archive["numbers"]
    frames = []
    for frame_positions, lengths in zip(positions, cell_lengths):
        atoms = Atoms(numbers=numbers, positions=frame_positions, cell=np.diag(lengths), pbc=True)
        frames.append(atoms.repeat(replicate))
    return frames


def initial_positions(atoms, init: str) -> np.ndarray:
    """test47: starting configuration for generation. 'crystal' matches test33/test45/test46
    (denoise the ideal structure itself); 'random' places atoms uniformly at random in the cell,
    which is the actual test of whether a crystal can be generated from nothing."""
    if init == "crystal":
        return np.asarray(atoms.positions, dtype=np.float32)
    if init == "random":
        lengths = np.diag(np.asarray(atoms.cell))
        return (np.random.rand(len(atoms), 3) * lengths).astype(np.float32)
    raise ValueError(f"Unknown --init: {init}")


def loss_fn(model: nn.Module, data: Data) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(data), data.dx)


def run_epoch(loader, model, optimizer, device, rattle, downselect, train: bool) -> float:
    model.train(train)
    total = 0.0
    for data in loader:
        data = data.to(device)
        data = rattle(data)
        data = downselect(data)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, data)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = loss_fn(model, data)
        total += loss.item()
    return total / len(loader)


# =============================================================================
# Training -- resumable via a restart file (optimizer state, epoch, loss
# history, RNG state). The FINAL checkpoint (args.checkpoint) is a plain
# state-dict, matching every other script in this project
# (model.load_state_dict(torch.load(checkpoint_path))) -- the restart file is
# a separate, internal-only artifact removed once training completes.
# =============================================================================


def train_model(
    args: argparse.Namespace,
    reference_atoms_list: list,
    model: nn.Module,
    device: torch.device,
    deadline: Optional[float],
    restart_path: Path,
    output_dir: Path,
) -> bool:
    dataset = PeriodicStructureDataset(reference_atoms_list, large_cutoff=args.large_cutoff, duplicate=args.duplicate)
    num_train = int(args.train_ratio * len(dataset))
    num_valid = len(dataset) - num_train
    train_set, valid_set = torch.utils.data.random_split(dataset, [num_train, num_valid])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    num_samples = len(dataset)
    train_epochs = max(1, int(args.num_updates / (num_samples / args.batch_size)))
    log(output_dir, f"{train_epochs} epochs -> ~{args.num_updates} optimizer updates "
        f"({num_samples} samples = {len(reference_atoms_list)} reference structure(s) x "
        f"{args.duplicate} duplicates, batch {args.batch_size}).")

    rattle = RattleParticles(sigma_max=args.sigma_max)
    downselect = DownselectEdges(cutoff=args.cutoff)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learn_rate)

    start_epoch = 0
    train_losses: list[float] = []
    valid_losses: list[float] = []
    if restart_path.exists():
        restart = torch_load(restart_path)
        model.load_state_dict(restart["model"])
        optimizer.load_state_dict(restart["optimizer"])
        start_epoch = int(restart["epoch"])
        train_losses = list(restart["train_losses"])
        valid_losses = list(restart["valid_losses"])
        restore_rng_state(restart.get("rng_state"))
        log(output_dir, f"Resuming training from epoch {start_epoch}/{train_epochs}.")

    for epoch in range(start_epoch, train_epochs):
        if deadline_reached(deadline):
            atomic_torch_save(
                {
                    "format_version": 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "train_epochs": train_epochs,
                    "rng_state": capture_rng_state(),
                },
                restart_path,
            )
            log(output_dir, f"Time budget reached; training restart saved at epoch {epoch}/{train_epochs}.")
            return False

        train_loss = run_epoch(train_loader, model, optimizer, device, rattle, downselect, train=True)
        valid_loss = run_epoch(valid_loader, model, optimizer, device, rattle, downselect, train=False)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if (epoch + 1) % args.training_checkpoint_epochs == 0 or epoch + 1 == train_epochs:
            atomic_torch_save(
                {
                    "format_version": 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "train_epochs": train_epochs,
                    "rng_state": capture_rng_state(),
                },
                restart_path,
            )
            log(output_dir, f"epoch {epoch + 1}/{train_epochs}  train={train_loss:.4f}  valid={valid_loss:.4f}")

    figure, (loss_ax, log_ax) = plt.subplots(1, 2, figsize=(10, 3.5))
    loss_ax.plot(train_losses, label="train")
    loss_ax.plot(valid_losses, label="valid")
    loss_ax.set(xlabel="Epoch", ylabel="Loss")
    loss_ax.legend()
    log_ax.semilogy(train_losses, label="train")
    log_ax.semilogy(valid_losses, label="valid")
    log_ax.set(xlabel="Epoch")
    log_ax.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss.png", dpi=160)
    plt.close(figure)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(model.state_dict(), args.checkpoint)
    if restart_path.exists():
        restart_path.unlink()
    log(output_dir, f"Training complete. Saved checkpoint: {args.checkpoint}")
    return True


# =============================================================================
# Generation -- annealed denoising then a no-noise polish pass, resumable via
# its own restart file. Graph built at args.cutoff throughout (see module
# docstring, fix #2) -- NOT args.large_cutoff, which is only used to build
# the wider pre-rattle graph during TRAINING dataset construction above.
# =============================================================================


@torch.no_grad()
def generate(
    args: argparse.Namespace,
    atoms,
    model: nn.Module,
    device: torch.device,
    restart_path: Path,
    deadline: Optional[float],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    output_dir = restart_path.parent
    numbers = np.asarray(atoms.numbers)
    si_mask_numpy = numbers == 14
    si_mask_device = torch.tensor(si_mask_numpy, dtype=torch.bool, device=device)
    species = torch.tensor(LabelEncoder().fit_transform(numbers), dtype=torch.long, device=device)
    cell = torch.tensor(np.asarray(atoms.cell), dtype=torch.float32, device=device)
    pbc = atoms.pbc

    phase = "noisy"
    noisy_index = 0
    polish_index = 0
    start_positions = initial_positions(atoms, args.init)  # test47: --init crystal|random
    current_positions = torch.tensor(start_positions, dtype=torch.float32, device=device)
    cg_frames = [np.asarray(start_positions[si_mask_numpy], dtype=np.float32)]

    if restart_path.exists():
        restart = torch_load(restart_path)
        expected = (args.gen_noisy_steps, args.gen_polish_steps, args.gen_max_sigma, args.cutoff, args.init)
        found = (restart["noisy_steps"], restart["polish_steps"], restart["max_sigma"], restart["cutoff"],
                 restart.get("init", "crystal"))
        if found != expected:
            raise RuntimeError(
                f"Restart settings do not match this invocation: restart={found}, requested={expected}. "
                "Use the same settings or select a new output directory."
            )
        phase = restart["phase"]
        noisy_index = int(restart["noisy_index"])
        polish_index = int(restart["polish_index"])
        current_positions = restart["current_positions"].to(device)
        cg_frames = [frame.numpy() for frame in restart["cg_frames_angstrom"]]
        restore_rng_state(restart.get("rng_state"))
        log(output_dir, f"Resuming generation phase={phase}, noisy={noisy_index}/{args.gen_noisy_steps}, "
            f"polish={polish_index}/{args.gen_polish_steps}")

    def save_restart(current_phase, noisy_i, polish_i):
        atomic_torch_save(
            {
                "format_version": 1,
                "phase": current_phase,
                "noisy_index": noisy_i,
                "polish_index": polish_i,
                "current_positions": current_positions.detach().cpu(),
                "cg_frames_angstrom": torch.from_numpy(np.stack(cg_frames)).float(),
                "noisy_steps": args.gen_noisy_steps,
                "polish_steps": args.gen_polish_steps,
                "max_sigma": args.gen_max_sigma,
                "cutoff": args.cutoff,
                "init": args.init,
                "rng_state": capture_rng_state(),
            },
            restart_path,
        )

    sigmas = (
        torch.linspace(args.gen_max_sigma, 0.001, args.gen_noisy_steps, device=device)
        if args.gen_noisy_steps else torch.empty(0, device=device)
    )

    def make_data() -> Data:
        return Data(x=species, pos=current_positions, cell=cell, pbc=pbc, numbers=numbers)

    if phase == "noisy":
        for index in range(noisy_index, args.gen_noisy_steps):
            if deadline_reached(deadline):
                save_restart("noisy", index, polish_index)
                log(output_dir, "Time budget reached; noisy-generation restart saved.")
                return None

            data = graph_on_device(make_data(), args.cutoff, numbers)
            displacement = model(data) + sigmas[index] * torch.randn_like(current_positions)
            current_positions = current_positions - displacement
            cg_frames.append(current_positions[si_mask_device].detach().cpu().numpy())
            noisy_index = index + 1

            if noisy_index % args.generation_checkpoint_steps == 0:
                save_restart("noisy", noisy_index, polish_index)
                log(output_dir, f"noisy step {noisy_index}/{args.gen_noisy_steps}")

        phase = "polish"
        save_restart(phase, noisy_index, polish_index)

    for index in range(polish_index, args.gen_polish_steps):
        if deadline_reached(deadline):
            save_restart("polish", noisy_index, index)
            log(output_dir, "Time budget reached; polish-generation restart saved.")
            return None

        data = graph_on_device(make_data(), args.cutoff, numbers)
        current_positions = current_positions - model(data)
        cg_frames.append(current_positions[si_mask_device].detach().cpu().numpy())
        polish_index = index + 1

        if polish_index % args.generation_checkpoint_steps == 0:
            save_restart("polish", noisy_index, polish_index)
            log(output_dir, f"polish step {polish_index}/{args.gen_polish_steps}")

    if restart_path.exists():
        restart_path.unlink()
    return current_positions.detach().cpu().numpy(), np.stack(cg_frames)


def save_results(
    output_dir: Path,
    atoms,
    final_positions: np.ndarray,
    cg_frames_angstrom: np.ndarray,
    args: argparse.Namespace,
    checkpoint_sha256: str,
) -> dict:
    numbers = np.asarray(atoms.numbers)
    cell_angstrom = np.asarray(atoms.cell)
    final_wrapped = wrap_positions(final_positions, cell_angstrom)
    final_atoms = Atoms(numbers=numbers, positions=final_wrapped, cell=cell_angstrom, pbc=atoms.pbc)
    ase.io.write(output_dir / "final_structure.extxyz", final_atoms)

    wrapped_cg = np.stack([wrap_positions(frame, cell_angstrom) for frame in cg_frames_angstrom])
    cg_positions_nm = (wrapped_cg / 10.0).astype(np.float32)
    cell_nm = (cell_angstrom / 10.0).astype(np.float32)
    np.save(output_dir / "positions.npy", cg_positions_nm)
    np.savez(
        output_dir / "manifest.npz",
        positions=cg_positions_nm,
        cell_nm=cell_nm,
        labels=np.full(cg_positions_nm.shape[0], 11, dtype=np.int32),  # 11 = 2x2x2-supercell DM2-generated crystal
        source="DM2 test47 checkpoint generation (2x2x2 supercell); Si-only CG trajectory",
        checkpoint_sha256=checkpoint_sha256,
    )

    reference_positions = wrap_positions(np.asarray(atoms.positions), cell_angstrom)
    generated_bonds, generated_angles = bond_and_angle_stats(final_wrapped, numbers, cell_angstrom)
    reference_bonds, reference_angles = bond_and_angle_stats(reference_positions, numbers, cell_angstrom)
    save_histogram(reference_bonds, generated_bonds, output_dir / "bond_comparison.png",
                    "Si-O distance (Angstrom)", "SiO2 crystal (2x2x2) Si-O bond length", 1.61)
    save_histogram(reference_angles, generated_angles, output_dir / "angle_comparison.png",
                    "O-Si-O angle (degree)", "SiO2 crystal (2x2x2) O-Si-O angle", 109.47)

    metrics = {
        "num_frames": int(cg_positions_nm.shape[0]),
        "num_cg_particles": int(cg_positions_nm.shape[1]),
        "reference_bond_mean_angstrom": float(reference_bonds.mean()),
        "reference_bond_std_angstrom": float(reference_bonds.std()),
        "generated_bond_mean_angstrom": float(generated_bonds.mean()),
        "generated_bond_std_angstrom": float(generated_bonds.std()),
        "reference_angle_mean_degree": float(reference_angles.mean()),
        "reference_angle_std_degree": float(reference_angles.std()),
        "generated_angle_mean_degree": float(generated_angles.mean()),
        "generated_angle_std_degree": float(generated_angles.std()),
        "gen_noisy_steps": args.gen_noisy_steps,
        "gen_polish_steps": args.gen_polish_steps,
        "cutoff_angstrom": args.cutoff,
        "replicate": list(args.replicate),
        "init": args.init,
        "checkpoint_sha256": checkpoint_sha256,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def requested_configuration(args: argparse.Namespace, checkpoint_sha256: str) -> dict:
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "cutoff_angstrom": args.cutoff,
        "gen_noisy_steps": args.gen_noisy_steps,
        "gen_polish_steps": args.gen_polish_steps,
        "gen_max_sigma": args.gen_max_sigma,
        "replicate": list(args.replicate),
        "seed": args.seed,
        "init": args.init,
    }


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    deadline = started + args.time_budget_hours * 3600.0 if args.time_budget_hours > 0 else None
    crystal_data_path = args.crystal_data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not crystal_data_path.is_file():
        raise FileNotFoundError(f"Crystal reference data not found: {crystal_data_path}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Submit to a GPU node or use --allow-cpu for a smoke test.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cpu_count = int(os.environ.get("PBS_NCPUS", os.cpu_count() or 1))
    torch.set_num_threads(max(1, cpu_count))

    log(output_dir, f"DM2 root: {DM2_ROOT}")
    log(output_dir, f"Crystal data: {crystal_data_path}")
    log(output_dir, f"Replicate: {tuple(args.replicate)}")
    log(output_dir, f"Checkpoint: {args.checkpoint}")
    log(output_dir, f"Output: {output_dir}")
    log(output_dir, f"Device: {device}")
    if device.type == "cuda":
        log(output_dir, f"GPU: {torch.cuda.get_device_name(0)}")

    unit_atoms = ase.io.read(crystal_data_path, format="lammps-data")
    crystal_atoms = unit_atoms.repeat(tuple(args.replicate))
    log(output_dir, f"Unit cell: {len(unit_atoms)} atoms {np.diag(np.asarray(unit_atoms.cell))} A -> "
        f"replicated: {len(crystal_atoms)} atoms {np.diag(np.asarray(crystal_atoms.cell))} A")

    model = build_model(args.cutoff, device, args.irreps_hidden, args.irreps_edge, args.num_convs)
    if args.checkpoint.is_file():
        model.load_state_dict(torch_load(args.checkpoint, map_location="cpu"))
        model.to(device)
        model.eval()
        log(output_dir, f"Found existing checkpoint, skipping training: {args.checkpoint}")
    else:
        log(output_dir, "No existing checkpoint -- training from scratch.")
        if args.single_reference or not args.reference_frames.is_file():
            if not args.single_reference:
                log(output_dir, f"--reference-frames not found ({args.reference_frames}); "
                    "falling back to the single ideal --crystal-data structure.")
            else:
                log(output_dir, "--single-reference passed: training on the single ideal "
                    "--crystal-data structure (test45/test46 behaviour).")
            reference_atoms_list = [crystal_atoms]
        else:
            reference_atoms_list = load_reference_frames(args.reference_frames, tuple(args.replicate))
            log(output_dir, f"Training reference: {len(reference_atoms_list)} thermal MD snapshots "
                f"from {args.reference_frames.name} (test47; test45/test46 used 1 ideal structure).")
        training_restart = output_dir / "training_restart.pt"
        complete = train_model(args, reference_atoms_list, model, device, deadline, training_restart, output_dir)
        if not complete:
            log(output_dir, "Submit the PBS job again with the same settings to continue training.")
            return 0
        model.eval()

    checkpoint_sha256 = file_sha256(args.checkpoint)
    configuration = requested_configuration(args, checkpoint_sha256)
    completion_path = output_dir / "run_complete.json"
    if completion_path.exists() and not args.force:
        completed = json.loads(completion_path.read_text(encoding="utf-8"))
        if completed.get("configuration") == configuration:
            log(output_dir, f"This configuration is already complete: {completion_path}")
            return 0
        log(output_dir, "Completed output uses different settings; starting a new generation.")

    remaining_budget = None
    if deadline is not None:
        remaining_budget = deadline - time.monotonic()
        if remaining_budget <= 120.0:
            log(output_dir, "No time budget remains for generation after training; resubmit to continue.")
            return 0

    log(output_dir, f"Generation: {args.gen_noisy_steps} noisy + {args.gen_polish_steps} polish steps; "
        f"cutoff={args.cutoff} Angstrom (see module docstring, fix #2 -- large_cutoff is NOT used here); "
        f"init={args.init}")

    result = generate(args, crystal_atoms, model, device, output_dir / "generation_restart.pt", deadline)
    if result is None:
        log(output_dir, "Submit the PBS job again with the same settings to continue generation.")
        return 0

    final_positions, cg_frames = result
    metrics = save_results(output_dir, crystal_atoms, final_positions, cg_frames, args, checkpoint_sha256)
    elapsed_hours = (time.monotonic() - started) / 3600.0
    completion = {
        "status": "complete",
        "elapsed_hours_this_submission": elapsed_hours,
        "configuration": configuration,
        "metrics": metrics,
    }
    completion_path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(output_dir, f"Completed in {elapsed_hours:.3f} hours. Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
