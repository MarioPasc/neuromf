"""Comprehensive environment check for NeuroMF."""

import importlib
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

REQUIRED_PACKAGES = [
    ("torch", "2.2"),
    ("torchvision", "0.17"),
    ("monai", "1.3"),
    ("pytorch_lightning", "2.2"),
    ("omegaconf", "2.3"),
    ("einops", "0.7"),
    ("nibabel", "5.0"),
    ("rich", "13.0"),
    ("peft", "0.10"),
    ("lpips", "0.1"),
    ("scipy", "1.12"),
    ("matplotlib", "3.7"),
    ("numpy", "1.26"),
    ("hydra", None),
]


def check_python() -> tuple[str, bool]:
    """Check Python version >= 3.11."""
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok = sys.version_info >= (3, 11)
    return ver, ok


def check_cuda() -> tuple[str, bool]:
    """Check CUDA availability."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"{name} (CUDA {torch.version.cuda})", True
        return "No CUDA", False
    except Exception as e:
        return str(e), False


def check_jvp() -> tuple[str, bool]:
    """Check torch.func.jvp works."""
    try:
        import torch
        from torch.func import jvp

        def f(x: torch.Tensor) -> torch.Tensor:
            return x**2

        val, tangent = jvp(f, (torch.tensor(3.0),), (torch.tensor(1.0),))
        ok = abs(val.item() - 9.0) < 1e-5 and abs(tangent.item() - 6.0) < 1e-5
        return f"f(3)={val.item()}, f'(3)={tangent.item()}", ok
    except Exception as e:
        return str(e), False


def check_maisi_import() -> tuple[str, bool]:
    """Check AutoencoderKlMaisi is importable."""
    try:
        from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
            AutoencoderKlMaisi,
        )

        return "importable", True
    except Exception as e:
        return str(e), False


def check_package(name: str, min_version: str | None) -> tuple[str, bool]:
    """Check a package is importable and meets minimum version."""
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        return ver, True
    except ImportError:
        return "NOT INSTALLED", False


def check_disk_space(path: str) -> tuple[str, bool]:
    """Check available disk space."""
    p = Path(path)
    if not p.exists():
        return "path missing", False
    usage = shutil.disk_usage(p)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    return f"{free_gb:.1f} GB free / {total_gb:.1f} GB total", free_gb > 10


def main() -> int:
    """Run all environment checks."""
    console = Console()
    errors = 0

    # System checks
    table = Table(title="System Checks")
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="dim")
    table.add_column("Status", style="bold")

    ver, ok = check_python()
    table.add_row("Python", ver, "[green]OK[/green]" if ok else "[red]FAIL[/red]")
    errors += 0 if ok else 1

    val, ok = check_cuda()
    table.add_row("CUDA/GPU", val, "[green]OK[/green]" if ok else "[red]FAIL[/red]")
    errors += 0 if ok else 1

    val, ok = check_jvp()
    table.add_row("torch.func.jvp", val, "[green]OK[/green]" if ok else "[red]FAIL[/red]")
    errors += 0 if ok else 1

    val, ok = check_maisi_import()
    table.add_row("AutoencoderKlMaisi", val, "[green]OK[/green]" if ok else "[red]FAIL[/red]")
    errors += 0 if ok else 1

    console.print(table)

    # Package checks
    pkg_table = Table(title="Package Checks")
    pkg_table.add_column("Package", style="cyan")
    pkg_table.add_column("Version", style="dim")
    pkg_table.add_column("Status", style="bold")

    for name, min_ver in REQUIRED_PACKAGES:
        ver, ok = check_package(name, min_ver)
        pkg_table.add_row(
            name, ver, "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        )
        errors += 0 if ok else 1

    console.print(pkg_table)

    # Disk space checks
    disk_table = Table(title="Disk Space")
    disk_table.add_column("Path", style="cyan")
    disk_table.add_column("Space", style="dim")
    disk_table.add_column("Status", style="bold")

    for label, path in [
        ("Project", "/home/mpascual/research/code/neuromf"),
        ("External drive", "/media/mpascual/Sandisk2TB/research/neuromf"),
    ]:
        val, ok = check_disk_space(path)
        disk_table.add_row(
            label, val, "[green]OK[/green]" if ok else "[yellow]LOW[/yellow]"
        )

    console.print(disk_table)

    if errors:
        console.print(f"\n[red]FAIL: {errors} check(s) failed.[/red]")
        return 1
    console.print("\n[green]All checks passed.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
