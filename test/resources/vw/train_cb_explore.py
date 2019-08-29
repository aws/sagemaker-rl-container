import argparse
import json
import os
from pathlib import Path
import logging

from vowpalwabbit import pyvw

from io_utils import JsonLinesReader, extract_model, CSVReader, validate_experience
from vw_utils import TRAIN_CHANNEL, MODEL_CHANNEL, save_vw_model, transform_to_vw

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

    vw_args_base = f"--cb_explore {num_arms} --quiet --epsilon 0"
#     vw_args_base = f"--cb_explore {num_arms} --quiet --bag 2"
#     vw_args_base = f"--cb_explore {num_arms} --quiet --cover 3"

    if TRAIN_CHANNEL not in channel_names:
        logging.info("No training data found. Saving a randomly initialized model!")
        vw_model = pyvw.vw(vw_args_base)
        save_vw_model(vw_model, vw_args_base)
    else:
        if MODEL_CHANNEL not in channel_names:
            logging.info(f"No pre-trained model has been specified in channel {MODEL_CHANNEL}."
                         f"Training will start from scratch.")
            vw_args = f"{vw_args_base}"
        else:
            # Load the pre-trained model for training.
            model_folder = os.environ[f'SM_CHANNEL_{MODEL_CHANNEL.upper()}']
            _, weights_path = extract_model(model_folder)
            logging.info(f"Loading model from {weights_path}")
            vw_args = f"{vw_args_base} -i {weights_path}"

        vw_model = pyvw.vw(vw_args)
        training_data_dir = Path(os.environ["SM_CHANNEL_%s" % TRAIN_CHANNEL.upper()])
        training_files = [i for i in training_data_dir.rglob("*") if i.is_file() and i.suffix == ".csv"]
        logging.info("Processing training data: %s" % training_files)

        data_reader = CSVReader(input_files=training_files)
        data_iterator = data_reader.get_iterator()

        count = 0
        for experience in data_iterator:
            is_valid = validate_experience(experience)
            if not is_valid:
                pass
            vw_instance = "%s:%s:%s | %s" % (experience["action"],
                                             1 - experience["reward"],
                                             experience["action_prob"],
                                             transform_to_vw(experience["observation"]))
            vw_model.learn(vw_instance)
            count += 1

        save_vw_model(vw_model, vw_args_base)
        logging.info(f"Model learned using {count} training experiences.")


if __name__ == '__main__':
    main()
