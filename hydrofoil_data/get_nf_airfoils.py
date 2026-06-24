"""List all airfoils in the NeuralFoil training database (via AeroSandbox)."""

from pathlib import Path

from aerosandbox import _asb_root


def get_airfoil_database_path() -> Path:
    """Return path to the AeroSandbox airfoil database directory."""
    return _asb_root / "geometry" / "airfoil" / "airfoil_database"


def list_airfoils() -> list[str]:
    """Return sorted list of all airfoil names in the database."""
    db_path = get_airfoil_database_path()
    return sorted(p.stem for p in db_path.glob("*.dat"))


def main() -> None:
    """Print, then save, the full AeroSandbox airfoil database listing."""
    names = list_airfoils()

    # Print summary and full list
    print(f"NeuralFoil training airfoil database: {len(names)} airfoils")
    print(f"Database path: {get_airfoil_database_path()}")
    print("-" * 60)
    for name in names:
        print(name)

    # Save list to the output data directory
    out_path = (
        Path(__file__).parent.parent / "output" / "data" / "nf_airfoils.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(names) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()