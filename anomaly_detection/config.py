import os
import logging

logger = logging.getLogger("PDataset-SDK")
logger.setLevel(logging.DEBUG)

stdout_logger = logging.StreamHandler()
stdout_logger.setFormatter(
    logging.Formatter(
        '[%(name)s:%(filename)s:%(lineno)d] - [%(process)d] - '
        '[%(funcName)s:%(lineno)d] - %(asctime)s - %(levelname)s - %(message)s'
    )
)

logger.addHandler(stdout_logger)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DATASET_PATH = f"{CURRENT_DIR}/dataset/Train_data.csv"
