import logging
import multiprocessing
import psutil

from vw_serving.sagemaker.exceptions import CustomerError


class BaseServerConfig(object):
    """
    Base class for the server config.
    Server config allows algorithms to configure at runtime
    """

    MULTI_RECORD_STRATEGY = "MULTI_RECORD"
    SINGLE_RECORD_STRATEGY = "SINGLE_RECORD"

    @property
    def number_of_workers(self):
        """
        Control the number of Gunicorn workers in scoring service. Default is number of CPU's
        :return: integer, number of workers
        """
        return multiprocessing.cpu_count()

    @property
    def max_concurrent_transforms(self):
        """Get maximum number of concurrent transform requests.

        The platform uses this value as a recommendation when executing a transform (batch inference) operation.
        Note, however, that the actual number of concurrent transform requests sent by the platform
        can be different (either lower or higher) as a result of configuration override supplied by caller.

        :return: (int) maximum number of concurrent transform requests
        """

        # By default recommend to send the same number of requests as number of workers
        return self.number_of_workers

    @property
    def timeout(self):
        """Control the Gunicorn timeout, affects response timeout as well as boot timeout.
        :return: timeout in seconds
        :rtype: int
        """
        return 30

    @property
    def prefork_load_model(self):
        """
        Control whether the model should be loaded before the worker process is forked.
        :return:
        """
        return False

    @property
    def eia_compatible(self):
        """Indicates whether the model can run inference on EIA.
        :return: True if the algorithm supports inference of EIA, False otherwise.
        :rtype: bool
        """
        return False

    @property
    def batch_strategy(self):
        """Get batch strategy for transform jobs.
        :return: (str) batch strategy
        """
        return self.MULTI_RECORD_STRATEGY


class ServerConfigUtils(object):

    @staticmethod
    def get_model_size_limited_num_workers(model_size, buffer_factor=1):
        """
        Return number of workers to use in parallel.
        This number is determined by available_memory / (buffer * model_size)
        Number of workers is bounded from 1 to number of CPU's
        :param model_size: size of uncompressed model in bytes
        :param buffer_factor: multiplicative weight to buffer memory model will use, defaults to 1
        :return: number of memory constrained workers
        :rtype: int
        """
        available_memory = psutil.virtual_memory().available
        logging.info("Model Size: %d bytes", model_size)

        try:
            estimated_num_workers = int(available_memory / (model_size * buffer_factor))
        except ZeroDivisionError:
            raise CustomerError(
                "Unable to load model. Model size is 0 bytes, "
                "indicating an incorrect model location or an empty file. "
                "Please check that the location of the model is correct.")

        num_cpu = multiprocessing.cpu_count()
        logging.info("Number of vCPUs: %d", num_cpu)

        nworkers = max(min(num_cpu, estimated_num_workers), 1)
        return nworkers
