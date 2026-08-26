"""Hydra wrapper for the dataset stage."""

import hydra

from data.load import main


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def hydra_main(config):
    main(config)


if __name__ == "__main__":
    hydra_main()
