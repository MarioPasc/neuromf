"""Verify all paths from configs/base.yaml exist or can be created."""

import sys
from pathlib import Path

from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

# Input paths that MUST exist
INPUT_KEYS = [
    "paths.fomo60k_root",
    "paths.maisi_vae_weights",
    "paths.maisi_diffusion_weights",
    "paths.conda_python",
]

# Output paths that should be created if missing
OUTPUT_KEYS = [
    "paths.results_root",
    "paths.latents_dir",
    "paths.generated_dir",
    "paths.figures_dir",
    "paths.logs_dir",
    "paths.training_checkpoints",
]


def main() -> int:
    """Check all paths and return 0 if all OK, 1 otherwise."""
    config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        return 1

    cfg = OmegaConf.load(config_path)
    console = Console()
    table = Table(title="NeuroMF Path Verification")
    table.add_column("Key", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Status", style="bold")

    errors = 0

    for key in INPUT_KEYS:
        path_str = OmegaConf.select(cfg, key)
        p = Path(path_str)
        if p.exists():
            table.add_row(key, path_str, "[green]EXISTS[/green]")
        else:
            table.add_row(key, path_str, "[red]MISSING[/red]")
            errors += 1

    for key in OUTPUT_KEYS:
        path_str = OmegaConf.select(cfg, key)
        p = Path(path_str)
        if p.exists():
            table.add_row(key, path_str, "[green]EXISTS[/green]")
        else:
            p.mkdir(parents=True, exist_ok=True)
            table.add_row(key, path_str, "[yellow]CREATED[/yellow]")

    console.print(table)

    if errors:
        console.print(f"\n[red]FAIL: {errors} input path(s) missing.[/red]")
        return 1
    console.print("\n[green]All paths verified.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
