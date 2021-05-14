import click

from src.entities.train_pipeline_params import (
    read_train_parameters
)


@click.command("train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    settings = read_train_parameters(config_path)
    # train_pipeline()


if __name__ == "__main__":
    train_pipeline_command()
