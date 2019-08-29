import logging
import os
import json
import jsonschema
import uuid

import sys

from vw_serving.sagemaker.exceptions import CustomerError, PlatformError, PlatformKeyError, PlatformValueError
import vw_serving.sagemaker.config.environment as environment

BASE_DIR = os.getenv(environment.SAGEMAKER_DATA_PATH, "/opt/ml")

# Input training and evaluation data
INPUT_VOLUME = None
CONFIG_DIR = "config"

# Configuration file locations and parameter definitions
TRAIN_PARAMS_FILE_NAME = "hyperparameters.json"
RESOURCE_CONFIG_FILE_NAME = "resourceconfig.json"
DATA_CHANNEL_CONFIG_FILE_NAME = "inputdataconfig.json"
INIT_STATE_FILE_NAME = "state"

# temp data files
INPUT_DATA_DIR = "data"
TRAIN_DATA_DIR = "train"
VALIDATION_DATA_DIR = "validation"

# Logging location
LOG_VOLUME = '/logs'
LOG_FILE_NAME = 'training.log'

# Placing a stop file in /control will stop training
CONTROL_VOLUME = None
STOP_CONTROL_FILE_NAME = 'stop'

# Persistent Docker Volume to which the model artifacts will be saved
ARTIFACTS_VOLUME = None
MODEL_ARTIFACT_FILE_NAME = 'model'
STATE_ARTIFACT_FILE_NAME = 'state'

# Persistent Docker Volume to which a "failed" or "success" file will be written.
OUTPUT_VOLUME = None

# File that will be written to RESULT_VOLUME if training fails.
FAILURE_RESULT_FILE_NAME = "failure"

ERROR_VOLUME = None
ERROR_FILE_NAME = 'errors.log'

ERROR_FILE_PATH = None
INIT_STATE_FILE_PATH = None

TRAIN_CONFIG_FILE_PATH = None
RESOURCE_CONFIG_FILE_PATH = None
DATA_CHANNEL_CONFIG_FILE_PATH = None

# temporarily dumping evaluation data to a file in the artifact dir
TEMP_EVAL_ARTIFACT_PATH = None
STOP_CONTROL_FILE_PATH = None

ERROR_LOGGER_ID = 'error'

_config = None
_data_channel = None
_setup_loggers = False


def initialize_path_constants():
    global INPUT_VOLUME
    global CONTROL_VOLUME
    global ARTIFACTS_VOLUME
    global OUTPUT_VOLUME
    global ERROR_VOLUME
    global ERROR_FILE_PATH
    global INIT_STATE_FILE_PATH
    global TRAIN_CONFIG_FILE_PATH
    global RESOURCE_CONFIG_FILE_PATH
    global DATA_CHANNEL_CONFIG_FILE_PATH
    global TEMP_EVAL_ARTIFACT_PATH
    global STOP_CONTROL_FILE_PATH

    INPUT_VOLUME = os.path.join(BASE_DIR, "input")
    CONTROL_VOLUME = os.path.join(BASE_DIR, 'control')
    ARTIFACTS_VOLUME = os.path.join(BASE_DIR, "model")
    OUTPUT_VOLUME = os.path.join(BASE_DIR, "output")
    ERROR_VOLUME = os.path.join(BASE_DIR, "errors")

    ERROR_FILE_PATH = os.path.join(ERROR_VOLUME, ERROR_FILE_NAME)
    INIT_STATE_FILE_PATH = os.path.join(INPUT_VOLUME, INIT_STATE_FILE_NAME)

    TRAIN_CONFIG_FILE_PATH = os.path.join(INPUT_VOLUME, CONFIG_DIR, TRAIN_PARAMS_FILE_NAME)
    RESOURCE_CONFIG_FILE_PATH = os.path.join(INPUT_VOLUME, CONFIG_DIR, RESOURCE_CONFIG_FILE_NAME)
    DATA_CHANNEL_CONFIG_FILE_PATH = os.path.join(INPUT_VOLUME, CONFIG_DIR, DATA_CHANNEL_CONFIG_FILE_NAME)

    # temporarily dumping evaluation data to a file in the artifact dir
    TEMP_EVAL_ARTIFACT_PATH = os.path.join(OUTPUT_VOLUME, "data", "evaluation")
    STOP_CONTROL_FILE_PATH = os.path.join(CONTROL_VOLUME, STOP_CONTROL_FILE_NAME)


initialize_path_constants()


class StopTrain(Exception):
    pass


class JsonSchemaValidationError(CustomerError):
    def __init__(self, e):
        """
        :param e: (jsonschema.ValidationError) parent exception instance
        """
        super(JsonSchemaValidationError, self).__init__(
            message=ConfigValidator.pretty_validation_message(e), caused_by=e
        )

    def public_failure_message(self):
        return self.get_error_summary()


class ConfigValidator:
    def __init__(self, schema_file):
        with open(schema_file) as sf:
            self.schema = json.load(sf)

    def validate(self, config):
        """
        Validates that config contains the required list of parameter keys.
        :param config: the configuration dictionary to validate.
        :return:
        """
        # TODO: Implement validation
        pass
        # try:
        #     resolver = jsonschema.RefResolver(base_uri="file://" + BASE_SCHEMA_PATH + "/", referrer=self.schema)
        #     jsonschema.validate(config, self.schema, resolver=resolver)
        # except jsonschema.ValidationError as e:
        #     raise JsonSchemaValidationError(e)

    @staticmethod
    def pretty_number_validator(validator):
        """Returns a human friendly description of the number/integer validator or None if not applicable"""
        allowed_keys = {'type', 'minimum', 'maximum'}
        if validator['type'] in ['number', 'integer'] and set(validator.keys()).issubset(allowed_keys):
            article = "an" if validator['type'] == 'integer' else 'a'
            message = "{} {}".format(article, validator['type'])
            if 'minimum' in validator and 'maximum' in validator:
                message += " between {} and {}".format(validator['minimum'], validator['maximum'])
            elif 'minimum' in validator:
                message += " greater than {}".format(validator['minimum'])
            elif 'maximum' in validator:
                message += " less than {}".format(validator['maximum'])
            else:
                return None
            return message
        else:
            return None

    @staticmethod
    def pretty_string_validator(validator):
        """Returns a human friendly description of the string validator or None if not applicable"""
        allowed_keys = {'type', 'pattern'}
        if validator['type'] == 'string' and set(validator.keys()).issubset(allowed_keys):
            message = "a string"
            if 'pattern' in validator:
                message += " which matches the pattern '{}'".format(validator['pattern'])
            return message
        else:
            return None

    @staticmethod
    def pretty_fallback_message(e):
        try:
            return "The value '{}' is not valid for '{}'. Reason: {}".format(e.instance, ".".join(e.path), str(e))
        except Exception:
            return str(e)

    @staticmethod
    def pretty_validation_message(e):
        try:
            if e.validator == 'enum':
                return ("The value '{}' is not valid for the '{}' "
                        "hyperparameter which accepts one of the following: {}").format(
                    e.instance, ".".join(e.path), ", ".join(map(lambda v: "'{}'".format(v), e.validator_value))
                )
            elif e.validator == 'type':
                return "The value '{}' is not valid for the '{}' hyperparameter which expects a {}".format(
                    e.instance, ".".join(e.path), e.validator_value
                )
            elif e.validator == 'additionalProperties':
                return e.message.replace('properties', 'hyperparameters')
            elif e.validator == 'required':
                # This is hacky since jsonschema doesn't actually expose which value was missing
                # so we have to determine it manually.
                missing_values = ["'{}'".format(key) for key in e.validator_value if key not in e.instance]
                return "No value(s) were specified for {} which are required hyperparameter(s)".format(
                    ", ".join(missing_values)
                )
            elif e.validator == 'oneOf':
                # oneOf validators are used to specify numbers.
                # For every number we have both a string and either a number or float entry.
                validator_messages = []
                for validator in e.validator_value:
                    potential_messages = list(filter(lambda v: v is not None,
                                                     [
                                                         ConfigValidator.pretty_string_validator(validator),
                                                         ConfigValidator.pretty_number_validator(validator)
                                                     ]))
                    if len(potential_messages) == 1:
                        validator_messages.append(potential_messages[0])
                    else:
                        # Either multiple validators matched or none did in which case we don't support
                        # this so just fallback.
                        return ConfigValidator.pretty_fallback_message(e)
                return "The value '{}' is not valid for the '{}' hyperparameter which expects one of the " \
                       "following: {}".format(e.instance, ".".join(e.path), "; or ".join(validator_messages))
            else:
                # Provide the default value.
                return ConfigValidator.pretty_fallback_message(e)
        except Exception:
            return ConfigValidator.pretty_fallback_message(e)


def setup_logging(log_file_name=None, log_level="info"):
    """
    Sets up loggers & log formatters.

    Configures file logging to log_file_name or stream logging to sys.stderr
    if log_file_name is None.

    If the /opt/ml/errors/ path exists, configures the error logger with
    file handler pointing to /opt/ml/errors/errors.log.
    Otherwise set it to stderr.

    Args:
        log_file_name: (str) file name to write public log records to, or None to write to stdout
        log_level: (str) log level for the public logger
    """
    global _setup_loggers
    if _setup_loggers:
        logging.warning("Loggers have already been setup.")
        return

    _setup_loggers = True

    # setup default logger
    log_formatter = logging.Formatter(
        '[%(asctime)s %(levelname)s %(thread)d] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')
    default_logger = logging.getLogger()
    # Don't fail if the log level is not known
    try:
        default_logger.setLevel(getattr(logging, log_level.upper()))
    except Exception:
        default_logger.setLevel(logging.INFO)
        default_logger.warning("Failed to set debug level to %s, using INFO", log_level)

    if log_file_name:
        handler = logging.FileHandler(log_file_name)
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(log_formatter)
    default_logger.addHandler(handler)

    # setting up trusted logger
    trusted_log_formatter = logging.Formatter(
        '[%(asctime)s %(levelname)s %(thread)d %(filename)s:%(lineno)d] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S')

    if os.path.isdir(ERROR_VOLUME):
        trusted_log_handler = logging.FileHandler(ERROR_FILE_PATH)
    else:
        trusted_log_handler = logging.StreamHandler(stream=sys.stderr)

    trusted_log_handler.setFormatter(trusted_log_formatter)
    trusted_log_handler.setLevel(logging.INFO)

    error_logger = get_error_logger()
    error_logger.addHandler(trusted_log_handler)
    error_logger.propagate = False


def get_configuration(default_conf, schema):
    """
    Returns a dictionary containing learning algorithm parameters.

    The returned configuration dictionary merges customer-specified parameters
    with default_conf & validates the schema.

    :param default_conf: pass the default config
    :param schema: schema to conform to
    :return: a dictionary of algorithm arguments
    """
    global _config
    if not _config:
        validator = ConfigValidator(schema)
        try:
            with open(default_conf, 'r') as def_conf_file:
                _config = json.load(def_conf_file)
                logging.info("Reading default configuration from %s: %s",
                             def_conf_file.name, _config)

            train_path = TRAIN_CONFIG_FILE_PATH
            with open(train_path, 'r') as train_conf_file:
                train_config = json.load(train_conf_file)
                logging.info("Reading provided configuration from %s: %s",
                             train_conf_file.name, train_config)

            _config.update(train_config)
        except IOError as e:
            logging.info("content of %s is %s", INPUT_VOLUME, os.listdir(INPUT_VOLUME))
            logging.info("content of %s is %s", os.path.join(INPUT_VOLUME, CONFIG_DIR),
                         os.listdir(os.path.join(INPUT_VOLUME, CONFIG_DIR)))
            raise PlatformError(message='Could not read configuration files from: {}'.format([default_conf,
                                                                                              TRAIN_CONFIG_FILE_PATH]),
                                caused_by=e)
        try:
            validator.validate(_config)
        except Exception:
            _config = None
            raise

    logging.info("Final configuration: %s", _config)
    return _config


def get_error_logger():
    """
    Returns the logger from logging for id ERROR_LOGGER_ID
    """
    return logging.getLogger(ERROR_LOGGER_ID)


def write_trusted_log_exception(private_error_message):
    """Write private exception message to the trusted error channel.

    For consistent behavior prefer using report_*() functions from error_handler.py
    """
    err_logger = get_error_logger()
    err_logger.exception(private_error_message)


def write_trusted_log_info(private_info_message):
    """
    Write private info message to the trusted log channel.
    """
    trusted_logger = get_error_logger()
    trusted_logger.info(private_info_message)


def write_failure_reason(failure_reason_text):
    """Write failure reason to /opt/ml/output/failure

    For consistent behavior prefer using report_*() functions from error_handler.py
    """
    with open(os.path.join(OUTPUT_VOLUME, FAILURE_RESULT_FILE_NAME), 'w') as f:
        f.write(failure_reason_text)


def report_failure(error_string=None):
    """Write the file "FAILURE" in the output volume and log the error with the error logger.

    For consistent behavior prefer using report_*() functions from error_handler.py

    :param error_string: (str) error message
    """
    write_failure_reason(error_string)
    write_trusted_log_exception(error_string)


def eval_batch_end_callback(*args, **kwargs):
    """
    Raises StopTrain if a file named STOP_CONTROL_FILE_NAME exists in CONTROL_DIR.
    """
    if os.path.isfile(STOP_CONTROL_FILE_PATH):
        raise StopTrain()


def get_artifact_file_path():
    return os.path.join(ARTIFACTS_VOLUME, "{}_{}".format(STATE_ARTIFACT_FILE_NAME, uuid.uuid4()))


def get_model_file_path(current_host):
    return os.path.join(ARTIFACTS_VOLUME, "{}_{}".format(MODEL_ARTIFACT_FILE_NAME, current_host))


# TODO: deprecate AWSALGO-1124
def get_train_input_dir():
    """
    :return: absolute path to train directory  /opt/ml/input/data/train if present
    """
    dc = get_data_config()
    return dc.get_path_from_channel_name("train")


# TODO: deprecate AWSALGO-1124
def get_validation_input_dir():
    """
    :return: absolute path to eval directory under /opt/ml/input/data/eval if present
    """
    dc = get_data_config()
    return dc.get_path_from_channel_name("validation")


def get_channel_dir(channel_name):
    """
    :return: absolute path to directory under /opt/ml/input/data/<channel_name> if present
    """
    dc = get_data_config()
    if dc and channel_name in dc.get_channel_names():
        return dc.get_path_from_channel_name(channel_name)
    return None


def get_files_recursively(path):
    file_list = []
    if path:
        for root, _, files in os.walk(path, topdown=False):
            if files:
                file_list.extend([os.path.join(root, f) for f in files])
    return file_list


def _read_host_config_from_file(file_path):
    try:
        with open(file_path) as f:
            cfg = json.load(f)
    except ValueError as e:
        raise PlatformValueError("The content of the resource configuration file is invalid.", caused_by=e)

    try:
        current_host = cfg['current_host']
    except KeyError as e:
        raise PlatformKeyError("The resource configuration file does not specify the current host name.", caused_by=e)

    peers = cfg.get('hosts', [])

    return current_host, set(peers)


def get_host_config(file_path=RESOURCE_CONFIG_FILE_PATH):
    """
    Return current host's dns and the dns of the peers in the cluster.
    :param file_path: path to the resource config
    :return: current host dns and the dns of the peers
    """
    global _current_host, _peers

    if not (_current_host or _peers):
        _current_host, _peers = _read_host_config(file_path)
    return _current_host, _peers


def _read_host_config(file_path):
    if not os.path.exists(file_path):
        logging.warning("The resource configuration file does not exists. Falling back to single machine.")
        return CURRENT_HOST_DNS, set(get_peers())

    return _read_host_config_from_file(file_path)


# TODO: Delete the rest of the file once the EASE deployment of 09/14 is complete.
CURRENT_HOST_DNS = os.getenv(environment.CURRENT_HOST, os.getenv(environment.ALGO_CONTAINER_NAME, "localhost"))
_current_host = None
_peers = None
# The env var containing the current container's dns
SAGEMAKER_HOSTS_LIST_SEPARATOR = ","


def get_peers():
    return os.getenv(environment.HOSTS, "localhost").split(SAGEMAKER_HOSTS_LIST_SEPARATOR)

def get_data_config():
    global _data_channel
    return _data_channel
