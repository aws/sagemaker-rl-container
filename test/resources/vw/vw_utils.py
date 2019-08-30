import os
import json

TRAIN_CHANNEL = "training"
EVAL_CHANNEL = "evaluation"
MODEL_CHANNEL = "pretrained_model"
MODEL_OUTPUT_DIR = os.environ.get('SM_MODEL_DIR', "/opt/ml/model")


def save_vw_metadata(meta):
    file_location = os.path.join(MODEL_OUTPUT_DIR, "vw.metadata")
    with open(file_location, "w") as f:
        f.write(meta)


def save_vw_model(model, meta):
    model.save(os.path.join(MODEL_OUTPUT_DIR, "vw.model"))
    save_vw_metadata(meta)


def transform_to_vw(x):
    x = json.loads(x)
    return " ".join(["%s:%s" % (i + 1, j) for i, j in enumerate(x)])
