import logging.config
import yaml
from src.entities.logging_params import LoggingParameters


def set_logging_config(log_param: LoggingParameters):
    with open(log_param.path_to_config, "r") as input_stream:
        config = yaml.safe_load(input_stream)
        logging.config.dictConfig(config)
