"""Deprecated compatibility entry point; use ``entity-clustering run`` instead."""

from pathlib import Path

from entity_clustering.runner import run_study


def main() -> None:
    root = run_study(Path("configs/studies/paper_v1.yaml"))
    print(f"Configured paper study completed: {root}")
    print("results.csv contains both vanilla and leave-one-out evaluation rows.")


if __name__ == "__main__":
    main()
