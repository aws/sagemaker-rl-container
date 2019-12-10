from __future__ import absolute_import
from io import StringIO
import json
import logging
import os
import signal
import sys
import warnings
import uuid
import datetime
import random

import numpy as np
from gunicorn.six import iteritems
from pkg_resources import iter_entry_points as iep
import flask
import gunicorn.app.base

try:
    import http.client as httplib  # python 3
except ImportError:
    import httplib  # python 2
import werkzeug

from vw_serving.utils import dynamic_import
from vw_serving.sagemaker.gpu import get_num_gpus
from vw_serving.sagemaker import integration as integ
from vw_serving.sagemaker.error_handler import report_batch_inference_sdk_error, report_online_inference_sdk_error
from vw_serving.sagemaker.exceptions import convert_to_algorithm_error, raise_with_traceback, CustomerError, \
    AlgorithmError
from vw_serving.sagemaker.config.server_config import BaseServerConfig
import vw_serving.sagemaker.config.environment as environment

from vw_serving.vw_agent import VWAgent

# TODO: Add metrics publishing
# from vw_serving.metrics import metrics_wrapper
# from vw_serving.metrics.metrics import MetricsFactory

CONTENT_TYPE_JSON = 'application/json'
CONTENT_TYPE_JSONLINES = 'application/jsonlines'
CONTENT_TYPE_CSV = 'text/csv'
CONTENT_TYPE_RECORDIO = 'application/x-recordio-protobuf'

REDIS_PUBLISHER_CHANNEL = "EXPERIENCES"
KNOWN_CLI_ARGS = ['-r', '--resources', '-w']

MODEL_DIR = integ.ARTIFACTS_VOLUME


class InferenceCustomerError(CustomerError):
    def public_failure_message(self):
        return self.get_error_summary()


class InferenceAlgorithmError(CustomerError):
    def public_failure_message(self):
        return self.get_error_summary()


class GunicornApplication(gunicorn.app.base.Application):
    """Gunicorn application

    By extending base.Application, this class gets access to configuration via the
    environment variable GUNICORN_CMD_ARGS and command line. This env variable gives the flexibility
    to configure gunicorn per-endpoint.

    See: http://docs.gunicorn.org/en/stable/settings.html

    See: https://code.amazon.com/packages/Gunicorn/blobs/5ea7b077710253db3ed6676525090ec04c05e4b8/--/gunicorn/app/base.py#L158 # noqa: E501
    """

    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(GunicornApplication, self).__init__()

    def load_vars(self):
        """Load env and command line vars.

        This method has to be copied because options run from brazil are also passed to Gunicorn, such as
        '--resource' or '-r' which is required for local training and serving. This failed the parser, which was
        using parse_args(). Copied and modified from superclass.

        See: https://code.amazon.com/packages/Gunicorn/blobs/5ea7b077710253db3ed6676525090ec04c05e4b8/--/gunicorn/app/base.py#L137 # noqa: E501
        """

        def check_unknown_args(args):
            """Check arguments Gunicorn is not aware of, and remove the argument names
             and values the application is aware of. Fail if there are any arguments remaining.

            :param args: Remainder arguments from Gunicorn argparser
            """
            unrecognized_args = {arg for arg in args if arg.startswith("-") and arg not in KNOWN_CLI_ARGS}
            if unrecognized_args:
                raise AlgorithmError("Unrecognized variables used: {}".format(", ".join(unrecognized_args)))

        def set_config(args, cfg):
            for k, v in vars(args).items():
                if v is None:
                    continue
                if k == "args":
                    continue
                cfg.set(k.lower(), v)

        parser = self.cfg.parser()
        parsed_args = parser.parse_known_args()
        known_args = parsed_args[0]
        unknown_args = parsed_args[1]

        check_unknown_args(unknown_args)

        env_vars = self.cfg.get_cmd_args_from_env()
        if env_vars:
            env_args = parser.parse_args(env_vars)
            set_config(env_args, self.cfg)

        # Lastly, update the configuration with any command line
        # settings. Note that identical env vars will not be updated.
        set_config(known_args, self.cfg)

    def load_config(self):
        """Load configuration.

        Configuration is loaded from the options passed in the constructor,
        and then load arguments from the CLI and the environment variable GUNICORN_CMD_ARGS.
        Arguments in the environment variable take precedence.

        See http://docs.gunicorn.org/en/stable/settings.html
        """
        for key, value in iteritems(self.options):
            key = key.lower()
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)
        self.load_vars()

    def load(self):
        return self.application


class ScoringService(object):
    PORT = os.getenv(environment.SAGEMAKER_BIND_TO_PORT, "8080")

    # NOTE: 6 MB max content length
    MAX_CONTENT_LENGTH = os.getenv(environment.MAX_CONTENT_LENGTH, 6 * 1024 * 1024)

    BATCH_INFERENCE = os.getenv(environment.SAGEMAKER_BATCH, 'false') == 'true'

    EIA_PRESENT = os.getenv(environment.SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT, 'false') == 'true'

    DEFAULT_INVOCATIONS_ACCEPT = os.getenv(environment.SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT, "")

    LOG_INFERENCE_DATA = os.getenv(environment.LOG_INFERENCE_DATA, 'true').lower() == 'true'

    app = flask.Flask(__name__)
    request_iterators = {}
    response_encoders = {}

    _model_class = None
    _model_id = None
    _server_config = None
    _model = None
    _redis_client = None

    @classmethod
    def _report_sdk_error(cls, sdk_error):
        """Report SDK error.

        Use different mechanisms for online inference and batch inference.
        """
        if cls.BATCH_INFERENCE:
            report_batch_inference_sdk_error(sdk_error)
        else:
            report_online_inference_sdk_error(sdk_error)

    @classmethod
    def _load_class_entry_point(cls, group, name):
        entry_points = tuple(iep(group=group, name=name))
        if not entry_points:
            return None

        class_entry_point = entry_points[0]
        loaded_class = class_entry_point.load()
        cls.app.logger.info("loaded entry point class %s:%s", group, class_entry_point.name)
        return loaded_class

    @classmethod
    def eia_enabled(cls):
        return cls._server_config.eia_compatible and cls.EIA_PRESENT and get_num_gpus() <= 0

    @classmethod
    def _load_pre_worker_entry_points(cls):
        try:
            server_config_class = cls._load_class_entry_point("algorithm.serve.server_config", "config_api")
            if server_config_class:
                cls._server_config = server_config_class()
        except Exception:
            cls.app.logger.exception("Unable to load server_config entry point")

        if cls._server_config is None:
            cls._server_config = BaseServerConfig()

        cls.app.logger.info("loading entry points")
        for entry_point in iep(group="algorithm.io.data_handlers.serve"):
            cls.request_iterators[entry_point.name] = entry_point.load()
            cls.app.logger.info("loaded request iterator %s", entry_point.name)

        for entry_point in iep(group="algorithm.request_iterators"):
            warnings.warn("entrypoint algorithm.request_iterators is deprecated "
                          "in favor of algorithm.io.data_handlers.serve", DeprecationWarning)
            cls.request_iterators[entry_point.name] = entry_point.load()
            cls.app.logger.info("loaded request iterator %s", entry_point.name)

        for entry_point in iep(group="algorithm.response_encoders"):
            cls.response_encoders[entry_point.name] = entry_point.load()
            cls.app.logger.info("loaded response encoder %s", entry_point.name)

        try:
            cls._model_class = cls._load_class_entry_point("algorithm", "model")
        except Exception as e:
            raise_with_traceback(InferenceAlgorithmError("Unable to load algorithm.model entry point", caused_by=e))

    @classmethod
    def get_model(cls):
        if cls._model is None:
            try:
                import redis
                redis_client = redis.Redis()
                cls._model_id = redis_client.get("model_id").decode()
                model_weights_loc = redis_client.get("{}:weights".format(cls._model_id)).decode()
                model_metadata_loc = redis_client.get("{}:metadata".format(cls._model_id)).decode()
                cls._model = VWAgent.load_model(metadata_loc=model_metadata_loc,
                                                weights_loc=model_weights_loc,
                                                test_only=True,
                                                quiet_mode=True)
                cls.app.logger.info(f"Loaded weights successfully for Model ID:{cls._model_id}")
            except Exception as e:
                raise_with_traceback(InferenceCustomerError("Unable to load model", caused_by=e))
        return cls._model

    @classmethod
    def _get_server_config(cls):
        if not cls._server_config:
            cls._load_pre_worker_entry_points()
        return cls._server_config

    @classmethod
    def get_transform_configuration(cls):
        """Get transform (batch inference) configuration.

        :return: (dict) a dictionary with two entries:
          max_concurrent_transforms: (int) number of concurrent transform requests to send.
            Platform can use this value to send appropriate number of concurrent requests to the container.
          max_payload_size: (int) maximum size of payload on /invocation request in bytes.
        """
        server_config = cls._get_server_config()

        return {
            'max_concurrent_transforms': server_config.max_concurrent_transforms,
            'batch_strategy': server_config.batch_strategy,
            'max_payload_size': cls.MAX_CONTENT_LENGTH,
        }

    @staticmethod
    def _post_worker_init(worker):
        """
        Gunicorn server hook http://docs.gunicorn.org/en/stable/settings.html#post-worker-init
        :param worker:
        """
        # Model is being loaded per worker because each worker communicates through PIPE with the VW C++ CLI
        try:
            if ScoringService.LOG_INFERENCE_DATA:
                import redis
                if ScoringService._redis_client is None:
                    ScoringService._redis_client = redis.Redis()
                    ScoringService.app.logger.info("Initiated redis client!")
        except Exception as e:
            sdk_error = convert_to_algorithm_error(e)
            ScoringService._report_sdk_error(sdk_error)
            sys.exit(sdk_error.exit_code)

        try:
            ScoringService.get_model()
            ScoringService._model.start()
        except Exception as e:
            sdk_error = convert_to_algorithm_error(e)
            ScoringService._report_sdk_error(sdk_error)
            sys.exit(sdk_error.exit_code)

    @staticmethod
    def _worker_exit(server, worker):
        """Do not cleanup resources on exit when memory profiler is enabled.

        Memory profiler imports multiprocessing module which causes exceptions on exit
        when used in the same process with fork().

        See:
          https://github.com/benoitc/gunicorn/issues/1391
          https://stackoverflow.com/questions/37692262
        """
        if os.getenv(environment.ENABLE_PROFILER):
            os._exit(0)

    @classmethod
    def parse_content_type(cls, content_type):
        content_type = content_type.lower() if content_type else CONTENT_TYPE_JSON

        tokens = content_type.split(";")
        content_type = tokens[0].strip()
        parameters = {}
        for token in tokens[1:]:
            key, value = token.split("=")
            key = key.strip()
            value = value.strip()
            parameters[key] = value

        if "shape" in parameters:
            parameters["shape"] = [int(s_i) for s_i in parameters["shape"].split(",")]

        return content_type, parameters

    @classmethod
    def get_num_workers(cls):
        forced_num_workers = int(os.getenv('NUM_WORKERS', 0))

        if forced_num_workers > 0:
            return forced_num_workers
        if cls.eia_enabled():
            return 1
        else:
            return int(cls._get_server_config().number_of_workers)

    @classmethod
    def _initialize(cls, daemon=False):
        cls._load_pre_worker_entry_points()

        # NOTE: Stop Flask application when SIGTERM is received as a result of "docker stop" command.
        signal.signal(signal.SIGTERM, cls.stop)

        nworkers = cls.get_num_workers()
        timeout = int(cls._get_server_config().timeout)

        cls.app.config["MAX_CONTENT_LENGTH"] = cls.MAX_CONTENT_LENGTH
        gunicorn_options = {
            "bind": "{}:{}".format("0.0.0.0", cls.PORT),
            "workers": min(nworkers, 4),
            "timeout": timeout,
            "worker_exit": cls._worker_exit,
            "pidfile": environment.PIDFILE,
            "daemon": daemon
        }
        # If prefork is chosen, call the model loading function immediately, otherwise register the model
        # loading function in the options map, os that the worker processes can call it after being forked.
        # if cls._server_config.prefork_load_model or True:
        # cls._post_worker_init(None)
        # else:

        # pre-fork model
        # cls._post_worker_init(None)

        gunicorn_options["post_worker_init"] = cls._post_worker_init
        cls.app.logger.info("Number of server workers: %s", nworkers)

        return gunicorn_options

    @classmethod
    def start(cls, daemon=False):
        integ.setup_logging()
        integ.write_trusted_log_info("worker started")

        try:
            options = cls._initialize(daemon=daemon)
        except Exception as e:
            sdk_error = convert_to_algorithm_error(e)
            ScoringService._report_sdk_error(sdk_error)
            sys.exit(sdk_error.exit_code)

        GunicornApplication(cls.app, options).run()

    @staticmethod
    def stop(*args, **kwargs):
        integ.write_trusted_log_info("worker closed")
        ScoringService.app.shutdown()


@ScoringService.app.errorhandler(httplib.INTERNAL_SERVER_ERROR)
def internal_server_error(e):
    sdk_error = convert_to_algorithm_error(e)
    ScoringService._report_sdk_error(sdk_error)
    return flask.Response(response="Internal Server Error", status=httplib.INTERNAL_SERVER_ERROR)


@ScoringService.app.route("/ping", methods=["GET"])
def ping():
    # TODO: implement health checks
    return flask.Response(status=httplib.OK)


def _error_predicate(return_value, exception):
    status_code_is_not_2xx = return_value and return_value.status_code / 100 != 2
    return exception or status_code_is_not_2xx


def _score_json(model, data):
    """

    Parameters
    ----------
    model : A reference to the scoring model object
    data : dict
        A dict consisting of request data

    Returns
    -------
    str
        A JSON encoded response string
    """
    shared_context = data.get("shared_context", None)
    actions_context = data.get("actions_context", None)
    top_k = int(data.get("top_k", 1))
    user_id = int(data.get("user_id", 0))

    event_id = uuid.uuid1().int
    dt = datetime.datetime.now()
    timestamp = int(dt.strftime("%s"))

    top_k_action_indices, action_probs = model.choose_actions(user_embedding=shared_context,
                                                              candidate_embeddings=actions_context,
                                                              top_k=top_k, user_id=user_id)
    sample_prob = random.uniform(0.0, 1.0)

    response_payload = json.dumps({"actions": top_k_action_indices,
                                   "action_probs": action_probs,
                                   "event_id": event_id,
                                   "timestamp": timestamp,
                                   "sample_prob": sample_prob,
                                   "model_id": ScoringService._model_id},
                                  ensure_ascii=False)

    blob_to_log = json.dumps({"actions": top_k_action_indices,
                              "action_probs": action_probs,
                              "event_id": event_id,
                              "shared_context": shared_context,
                              "actions_context": actions_context,
                              "timestamp": timestamp,
                              "model_id": ScoringService._model_id,
                              "sample_prob": sample_prob,
                              "type": "actions"},
                             ensure_ascii=False)

    if ScoringService.LOG_INFERENCE_DATA:
        # TODO: Log state, action, eventID
        ScoringService._redis_client.publish(REDIS_PUBLISHER_CHANNEL, blob_to_log)
    return response_payload


@ScoringService.app.route("/invocations", methods=["POST"])
def invocations():
    content_type, content_parameters = ScoringService.parse_content_type(flask.request.content_type)
    if content_type not in [CONTENT_TYPE_JSON, CONTENT_TYPE_JSONLINES]:
        sdk_error = InferenceCustomerError("Content-type {} not supported".format(content_type))
        ScoringService._report_sdk_error(sdk_error)
        return flask.Response(
            response="content-type {} not supported".format(content_type), status=httplib.UNSUPPORTED_MEDIA_TYPE
        )

    payload = flask.request.data
    if len(payload) == 0:
        return flask.Response(response="", status=httplib.NO_CONTENT)

    try:
        model = ScoringService.get_model()
    except Exception as e:
        sdk_error = convert_to_algorithm_error(e)
        ScoringService._report_sdk_error(sdk_error)
        return flask.Response(response="unable to load model", status=httplib.INTERNAL_SERVER_ERROR,
                              mimetype="application/json", content_type="application/json")

    if content_type == CONTENT_TYPE_JSON:
        data = json.loads(payload.decode("utf-8"))
        request_type = data.get("request_type", "observation").lower()

        if request_type == "reward":
            event_id = data["event_id"]
            reward = data["reward"]
            blob_to_log = {"event_id": int(event_id), "reward": float(reward), "type": "rewards"}
            blob_to_log = json.dumps(blob_to_log)
            if ScoringService.LOG_INFERENCE_DATA:
                ScoringService._redis_client.publish(REDIS_PUBLISHER_CHANNEL, blob_to_log)
                status = "success"
            else:
                status = "failure"
            return flask.Response(response='{"status": "%s"}' % status, status=httplib.OK)

        elif request_type == "model_id":
            model_info_payload = json.dumps({"model_id": ScoringService._model_id,
                                             "soft_model_update_status": "TBD: To be used for indicating rollbacks"})
            return flask.Response(response=model_info_payload, status=httplib.OK, mimetype="application/json",
                                  content_type="application/json")

        else:
            try:
                response_payload = _score_json(model, data)
                return flask.Response(response=response_payload, status=httplib.OK, mimetype="application/json",
                                  content_type="application/json")
            except Exception as e:
                return flask.Response(response=f"Incorrect JSON format error: {e}", status=httplib.BAD_REQUEST)

    else:
        #  Content type is application/jsonlines, which means this is Batch Inference mode
        data = payload.decode("utf-8")
        f = StringIO(data)
        response = [_score_json(model, json.loads(line)) for line in f.readlines()]
        response_payload = "\n".join(response)
        return flask.Response(response=response_payload, status=httplib.OK, mimetype="application/jsonlines",
                              content_type="application/jsonlines")


# TODO: Disabling this API temporarily as it is leading to some issues in local Batch Transform mode
@ScoringService.app.route("/execution-parameterss", methods=["GET"])
def execution_parameters():
    service_config = ScoringService.get_transform_configuration()
    try:
        parameters = {
            "MaxConcurrentTransforms": service_config["max_concurrent_transforms"],
            "BatchStrategy": service_config["batch_strategy"],
            "MaxPayloadInMB": service_config["max_payload_size"] // (1024 * 1024)  # convert bytes to MB
        }
    except KeyError as e:
        sdk_error = convert_to_algorithm_error(e)
        ScoringService._report_sdk_error(sdk_error)
        return flask.Response(response="unable to determine execution parameters", status=httplib.INTERNAL_SERVER_ERROR)

    response_text = json.dumps(parameters)
    return flask.Response(response=response_text, status=httplib.OK, mimetype="application/json")


if __name__ == "__main__":
    ScoringService.start()
