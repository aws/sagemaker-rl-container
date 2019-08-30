import os
import boto3
import redis
import time
import signal
import json
import logging
import subprocess
import shutil
import multiprocessing
from multiprocessing import Process
from pathlib import Path

from vw_serving.utils import dynamic_import, parse_s3_url, gen_random_string
import vw_serving.sagemaker.config.environment as environment
from vw_serving.sagemaker import integration as integ
from vw_serving.firehose_producer import FirehoseProducer
from vw_serving.sagemaker.exceptions import convert_to_algorithm_error, raise_with_traceback, AlgorithmError, CustomerError
from vw_serving.serve import REDIS_PUBLISHER_CHANNEL
from boto3.dynamodb.conditions import Key


logger = logging.getLogger()


class ModelDBClient(object):
    def __init__(self, table_session):
        self.table_session = table_session

    def check_model_record_exists(self, experiment_id, model_id):
        if self.get_model_record(experiment_id, model_id) is None:
            return False
        else:
            return True

    def get_model_record(self, experiment_id, model_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id) & Key('model_id').eq(model_id)
        )
        for i in response['Items']:
            return i
        return None


class ExperimentDBClient(object):
    def __init__(self, table_session):
        self.table_session = table_session

    def get_experiment_record(self, experiment_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id)
        )
        for i in response['Items']:
            return i
        return None


class ModelManager():
    def __init__(self):
        self.sagemaker_tar_gz = self._check_sagemaker_model_dir()
        self.experiment_id = os.getenv(environment.EXPERIMENT_ID, "default_experiment")
        self.model_id = os.getenv(environment.MODEL_ID, "default_model")

        self.poll_db = os.getenv(environment.MODEL_METADATA_POLLING, 'false').lower() == 'true'

        if self.poll_db:
            self._setup_boto_clients()

        self.log_inference_data = os.getenv(
            environment.LOG_INFERENCE_DATA, 'false').lower() == 'true'
        if self.log_inference_data:
            self.firehost_stream = os.getenv(environment.FIREHOSE_STREAM, None)
            if not self.firehost_stream:
                raise AlgorithmError(
                    f"Please specify a firehose stream as '{environment.FIREHOSE_STREAM}' environment variable.")

    def _check_ddb_table_existence(self, ddb_table_resource):
        # Ensure that the table exists. Throw an error otherwise.
        try:
            status = ddb_table_resource.table_status
            if status != "ACTIVE":
                raise AlgorithmError(
                    f"Table: '{ddb_table_resource.name}' is not an 'ACTIVE' table. Its status is {status}")
        except:
            raise AlgorithmError(
                    f"Table: '{ddb_table_resource.name}' does not exist in dynamo db")

    def _setup_boto_clients(self):
        aws_region = os.getenv("AWS_REGION", "us-west-2")
        self.boto_session = boto3.Session(region_name=aws_region)
        
        # initialize resource clients 
        self.exp_ddb_table = os.getenv(environment.EXP_METADATA_DYNAMO_TABLE, "")
        if not self.exp_ddb_table:
            raise AlgorithmError(
                    f"Please specify a dynamo db table name as '{environment.EXP_METADATA_DYNAMO_TABLE}' environment variable.")
        self.model_ddb_table = os.getenv(environment.MODEL_METADATA_DYNAMO_TABLE, "")
        if not self.model_ddb_table:
            raise AlgorithmError(
                    f"Please specify a dynamo db table name as '{environment.MODEL_METADATA_DYNAMO_TABLE}' environment variable.")

        self.exp_ddb_table_resource = self.boto_session.resource('dynamodb').Table(self.exp_ddb_table)
        self._check_ddb_table_existence(self.exp_ddb_table_resource)

        self.model_ddb_table_resource = self.boto_session.resource('dynamodb').Table(self.model_ddb_table)
        self._check_ddb_table_existence(self.model_ddb_table_resource)

        self.exp_ddb_wrapper = ExperimentDBClient(self.exp_ddb_table_resource)
        self.model_ddb_wrapper = ModelDBClient(self.model_ddb_table_resource)
        self.s3_resource = self.boto_session.resource('s3')

    def _check_sagemaker_model_dir(self):
        sagemaker_model_path = Path(integ.ARTIFACTS_VOLUME)
        if sagemaker_model_path.is_dir():
            model_files = list(sagemaker_model_path.rglob("*"))
            if len(model_files) >= 2:
                return True 
        return False

    def _restart_gunicorn_workers(self):
        if not os.path.isfile(environment.PIDFILE):
            raise AlgorithmError(
                "Gunicorn PID file not found. Perhaps the server is not running?")
        pid = int(open(environment.PIDFILE).read().strip())
        os.kill(pid, signal.SIGHUP)

    def _start_gunicorn_server(self):
        from vw_serving.serve import ScoringService
        server_process = Process(target=ScoringService.start, args=(False,))
        server_process.start()
        logger.info(f"Started server process with PID: {server_process.pid}")

    def _start_experience_logger(self):

        def start_firehose_producer():
            producer = FirehoseProducer(self.firehost_stream)
            producer.listen_to_redis_channel(channel=REDIS_PUBLISHER_CHANNEL)

        producer_process = Process(target=start_firehose_producer)
        producer_process.start()
        logger.info(
            f"Started producer process with PID: {producer_process.pid}")

    def _download_and_extract_model_tar_gz(self, model_id):
        """
        This function first gets the s3 location from dynamo db,
        downloads the model, extracts it and then
        returns a tuple (str, str) of metadata string and model weights URL on disk
        """
        deployable_model_id_record = self.model_ddb_wrapper.get_model_record(experiment_id=self.experiment_id,
                                                                             model_id=model_id)
        s3_uri = deployable_model_id_record.get("s3_model_output_path", "")
        if s3_uri:
            try:
                tmp_dir = Path(f"/opt/ml/downloads/{gen_random_string()}")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_model_tar_gz = os.path.join(tmp_dir.as_posix(), "model.tar.gz")
                bucket, key = parse_s3_url(s3_uri)
                self.s3_resource.Bucket(bucket).download_file(key, tmp_model_tar_gz)
                shutil.unpack_archive(filename=tmp_model_tar_gz, extract_dir=tmp_dir.as_posix())
                return self.get_model(tmp_dir.as_posix())
            except Exception as e:
                logger.exception(f"Could not parse or download {model_id} from {s3_uri} due to {e}")
                return None
        else:
            logger.exception(f"Could not s3 location of {model_id}")
            return None

    def _check_for_new_model(self):
        experiment_record = self.exp_ddb_wrapper.get_experiment_record(self.experiment_id)
        # if experiment_record.get("hosting_state", "") == "PENDING":
        next_model_to_host_id = experiment_record.get("hosting_workflow_metadata", {}).get("next_model_to_host_id", "")
        if next_model_to_host_id:
            if next_model_to_host_id != self.model_id:
                return next_model_to_host_id
            else:
                return None

    def get_model(self, disk_path=None, model_id=None):
        """
        Returns a tuple (str, str) of metadata string and model weights URL on disk
        """
        if disk_path:
            sagemaker_model_path = Path(disk_path)
            meta_files = list(sagemaker_model_path.rglob("vw.metadata"))
            if len(meta_files) == 0:
                raise CustomerError("'vw.metadata' not found in model files.")
            metadata_path = meta_files[0]

            model_files = list(sagemaker_model_path.rglob("vw.model"))
            if len(model_files) == 0:
                raise CustomerError("'vw.model' not found in model files.")
            model_path = model_files[0]
            return metadata_path.as_posix(), model_path.as_posix()
        elif model_id:
            return self._download_and_extract_model_tar_gz(model_id=model_id)

    def serve(self):
        if self.sagemaker_tar_gz:
            metadata_path, weights_path = self.get_model(disk_path=integ.ARTIFACTS_VOLUME)
        else:
            metadata_path, weights_path = self.get_model(model_id=self.model_id)

        redis_client = redis.Redis()
        redis_client.set("model_id", self.model_id)
        redis_client.set("{}:weights".format(self.model_id), weights_path)
        redis_client.set("{}:metadata".format(self.model_id), metadata_path)

        if self.log_inference_data:
            self._start_experience_logger()

        logger.info("Starting gunicorn...")
        self._start_gunicorn_server()
        logger.info("Started gunicorn.")
        sleep_seconds = 1
        if self.poll_db:
            while True:
                time.sleep(sleep_seconds)
                next_model_to_host_id = self._check_for_new_model()
                # logger.info("Fetching latest model from Dynamo")
                if next_model_to_host_id:
                    logger.info(
                        f"Found new model! Trying to replace Model ID: {self.model_id} with Model ID: {next_model_to_host_id}")

                    try:
                        metadata, weights = self.get_model(model_id=next_model_to_host_id)

                        # Delete the old model
                        redis_client.delete(self.model_id)
                        redis_client.delete("{}:weights".format(self.model_id))
                        redis_client.delete("{}:weights".format(self.model_id))

                        self.model_id = next_model_to_host_id
                        redis_client.set("model_id", self.model_id)
                        redis_client.set("{}:weights".format(self.model_id), weights)
                        redis_client.set("{}:metadata".format(self.model_id), metadata)
                        self._restart_gunicorn_workers()
                    except Exception as e:
                        logger.exception(f"Error happened when deploying model {next_model_to_host_id} due to {e}")
                        # TODO: Have a mechanism of testing if the model prediction works


def start_redis_server():
    ping = subprocess.Popen("redis-cli ping", shell=True,
                            stderr=subprocess.STDOUT,
                            stdout=subprocess.PIPE)
    stdout = ping.stdout.read()
    if stdout.decode().strip() == "PONG":
        logger.info("Redis server is already running.")
    else:
        p = subprocess.Popen("redis-server --bind 0.0.0.0 --loglevel warning", shell=True, stderr=subprocess.STDOUT)
        time.sleep(3)
        if p.poll() is not None:
            raise RuntimeError("Could not start Redis server.")
        else:
            logger.info("Redis server started successfully!")


def main():
    integ.setup_logging()
    start_redis_server()
    model_manager = ModelManager()
    model_manager.serve()


if __name__ == "__main__":
    main()
