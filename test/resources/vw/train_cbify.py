import argparse
import json
import os
from pathlib import Path
import logging
import subprocess

from vowpalwabbit import pyvw

from io_utils import JsonLinesReader, extract_model, CSVReader, validate_experience
from vw_utils import TRAIN_CHANNEL, MODEL_CHANNEL, save_vw_model, transform_to_vw, MODEL_OUTPUT_DIR

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    channel_names = json.loads(os.environ['SM_CHANNELS'])
    hyperparameters = json.loads(os.environ['SM_HPS'])
    num_arms = int(hyperparameters.get("num_arms", 0))
    if num_arms is 0:
        raise ValueError("Customer Error: Please provide a non-zero value for 'num_arms'")
    logging.info("channels %s" % channel_names)
    logging.info("hps: %s" % hyperparameters)

    vw_args_base = f"vw --cbify {num_arms} --cb_explore_adf --epsilon 0 -f {MODEL_OUTPUT_DIR}/vw.model"

    training_data_dir = Path(os.environ["SM_CHANNEL_%s" % TRAIN_CHANNEL.upper()])
    training_files = [i for i in training_data_dir.rglob("*") if i.is_file() and i.suffix == ".vw"]
    file_path = training_files[0].as_posix()
    logging.info("Processing training data: %s" % file_path)
    vw_args_base = f"{vw_args_base} -d {file_path}"
    proc = subprocess.Popen(vw_args_base, universal_newlines=False, shell=True)
    proc.wait()


if __name__ == '__main__':
    main()
