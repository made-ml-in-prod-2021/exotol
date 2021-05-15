import click
import logging
from src.log.utils import set_logging_config
from src.entities.train_pipeline_params import (
    read_train_parameters
)
from src.data.make_dataset import (
    read_data,
    train_val_split
)
from src.features.build_features import (
    build_transformer,
    create_features,
    create_target
)

from src.models.model_utils import (
    create_model
)
logger = logging.getLogger("pipeline")


def train_pipeline(settings):
    set_logging_config(settings.log_params)

    logger.info("Stage: read data")
    data = read_data(settings.input_data_path)

    logger.info("Stage: split data")
    train_df, valid_df = train_val_split(
        data,
        settings.split_params
    )

    logger.info("Stage: create train features")
    transformer = build_transformer(settings.feature_params)

    logger.info("Stage: create train features")
    train_features = create_features(
        transformer,
        train_df,
    )

    logger.info("Stage: create target")
    train_target = create_target(train_df, settings.feature_params)

    logger.info("Stage: create model")
    model = create_model(settings.model_params)

    # logger.info("Stage: train model")
    # train_model(model, train_features, train_target)
    #
    # logger.info("Stage: serialize model")
    # inference_pipeline = create_inference_pipeline(model, transformer)
    # serialize_model(inference_pipeline, settings.output_model_path)




@click.command("train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    settings = read_train_parameters(config_path)
    train_pipeline(settings)


if __name__ == "__main__":
    train_pipeline_command()
