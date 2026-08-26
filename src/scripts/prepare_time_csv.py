"""Command-line entry point for portable TIME dataset preparation."""

from data.time import main, prepare_time_csv

__all__ = ["main", "prepare_time_csv"]


if __name__ == "__main__":
    main()
