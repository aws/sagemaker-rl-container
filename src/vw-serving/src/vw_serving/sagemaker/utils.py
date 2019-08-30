import logging
import os
import socket
import time
from socket import gaierror
from subprocess import Popen, PIPE

from retrying import retry

from vw_serving.sagemaker.exceptions import PlatformError
import vw_serving.sagemaker.config.environment as environment


class Preconditions(object):
    CHECK_LENGTH_DEFAULT_ERROR_STRING = "Precondition check failed: length is zero."
    CHECK_NONE_DEFAULT_ERROR_STRING = "Precondition check failed: value is not None."
    CHECK_NOT_NONE_DEFAULT_ERROR_STRING = "Precondition check failed: value is None."
    CHECK_NOT_ALLOWED_VALUE_ERROR_STRING = "Precondition check failed: value is not one of the allowed values."
    CHECK_NOT_FLOAT_VALUE_ERROR_STRING = "Precondition check failed: value is not a float."

    @classmethod
    def check_length(cls, value, msg=None, exception_cls=ValueError):
        """
        Checks the length of the value. If length is zero, raises exception of type exception_cls.
        If the value is None, the exception raised by len() is re-raised as exception of type exception_cls.
        Otherwise returns the value.

        :param value: string or list value to be checked
        :param msg: the message to pass in the raised error. If None
        :param exception_cls: exception class
        :return: value or raises error when check fails.
        """

        try:
            if len(value) == 0:
                raise exception_cls(msg or cls.CHECK_LENGTH_DEFAULT_ERROR_STRING)
        except TypeError as err:
            raise exception_cls(msg or str(err))
        return value

    @classmethod
    def check_none(cls, value, msg=None, exception_cls=ValueError):
        """
        Checks if the value is None. If not None, raises exception of type exception_cls.

        :param value: value to be checked
        :param msg: the message to pass in the raised error, if None.
        :param exception_cls: exception class
        """

        if value is not None:
            raise exception_cls(msg or cls.CHECK_NONE_DEFAULT_ERROR_STRING)

    @classmethod
    def check_not_none(cls, value, msg=None, exception_cls=ValueError):
        """
        Checks if the value is None. If None, raises exception of type exception_cls.

        :param value: value to be checked
        :param msg: the message to pass in the raised error, if None.
        :param exception_cls: exception class
        :return: value or raises error when check fails.
        """

        if value is None:
            raise exception_cls(msg or cls.CHECK_NOT_NONE_DEFAULT_ERROR_STRING)
        return value

    @classmethod
    def check_allowed_value(cls, value, allowed_values, msg=None, exception_cls=ValueError):
        """
        Checks if the value is one of allowed_values. It raises exception of type exception_cls if
        allowed_values is not iterable or the value is not one of the allowed_values.

        :param value: value to be checked
        :param allowed_values: iterable of allowed values to be searched.
        :param msg: the message to pass in the raised error, if not found in the allowed_values.
        :param exception_cls: exception class
        :return: value or raises error when check fails.
        """
        try:
            if value in allowed_values:
                return value
            else:
                raise TypeError
        except TypeError:
            raise exception_cls(msg or cls.CHECK_NOT_ALLOWED_VALUE_ERROR_STRING)

    @classmethod
    def check_float(cls, value, msg=None, exception_cls=ValueError):
        try:
            float(value)
            return value
        except ValueError:
            raise exception_cls(msg or cls.CHECK_NOT_FLOAT_VALUE_ERROR_STRING)


def run_command(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    return p.returncode, out.rstrip(), err.rstrip()


def get_container_ip():
    """
    :return: Current host ip
    """
    return socket.gethostbyname(socket.gethostname())


# Max wait time of 15 minutes
# Max delay between response 0.5 sec.
@retry(wait_fixed=500, stop_max_delay=900000)
def lookup(dns):
    """
    Return the IP address (a string of the form '255.255.255.255') for the dns
    :param dns: dns to look up
    :return: IP address (a string of the form '255.255.255.255') for the dns
    """
    if os.getenv(environment.AI_ALGORITHMS_LOCAL_RUN):
        return socket.gethostbyname("localhost")
    return socket.gethostbyname(dns)


class MXDistributor(object):
    """
    Sets up the KVStore server, scheduler & sends train request to other containers
    """
    SCHEDULER_PORT = int(os.getenv(environment.SCHEDULER_PORT, 9000))

    def __init__(self, my_dns, peers, num_servers=1):
        self._my_dns = Preconditions.check_length(
            my_dns, msg="None or empty value provided for my_dns")
        self._peers = Preconditions.check_length(sorted(peers), msg="None or empty value provided for peers")
        self._scheduler_dns = Preconditions.check_length(
            self._peers[0], msg="None or empty value provided for scheduler_dns")
        self._num_instances = len(self._peers)
        try:
            self._scheduler_ip = lookup(self._scheduler_dns)
        except (IndexError, gaierror) as e:
            raise PlatformError(message="dns {} could not be resolved. Possibly, all nodes in the "
                                        "cluster are not available".format(self._scheduler_dns), caused_by=e)
        self._num_servers = max(min(num_servers, self._num_instances), 1)

        self.scheduler_pid = None
        self.server_pid = None

    @staticmethod
    def is_distributed(store_type):
        return store_type.startswith("dist_")

    def start(self):
        """
        starts the scheduler, server & workers

        :return:
        """
        if self._num_instances < 1:
            raise ValueError("Invalid learner instance count")

        # Run the parameter server scheduler and server processes. Only
        # one of the peers will run the scheduler process.
        if self._my_dns == self._scheduler_dns:
            self.scheduler_pid = self._run_parameter_server(role="scheduler")

        # current container is a server container
        if self._my_dns in self._get_server_dns_list():
            self.server_pid = self._run_parameter_server(role="server")

        # setup the env vars for the current process to run as a worker
        os.environ.update(self._get_dmlc_envs("worker"))
        logging.info("Environment: %s", os.environ.copy())

    def _run_parameter_server(self, role=None):
        """
        :param role: (str) process role
        :return: (int) shell process id
        """
        parameter_server_command = "python -c 'import mxnet'"
        parameter_server_env = os.environ.copy()
        parameter_server_env.update(self._get_dmlc_envs(role))
        logging.info("Launching parameter server for role %s", role)
        logging.info(repr(os.environ.copy()))
        logging.info("envs=%s", parameter_server_env)
        shell_process = Popen(parameter_server_command, shell=True, env=parameter_server_env)
        return shell_process.pid

    def _get_dmlc_envs(self, role):
        envs = {"DMLC_ROLE": role,
                "DMLC_PS_ROOT_URI": self._scheduler_ip,
                "DMLC_PS_ROOT_PORT": str(self.SCHEDULER_PORT),
                "DMLC_NUM_SERVER": str(self._num_servers),
                "DMLC_NUM_WORKER": str(self._num_instances)
                }
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            envs["PS_VERBOSE"] = "2"
        return envs

    def _get_server_dns_list(self):
        return self._peers[:self._num_servers]


class Stopwatch:
    """
    Stopwatch class, useful for timer metrics.

    Time is reported in milliseconds. Typical use:

    def main():
        timer = Stopwatch()
        thing_to_measure()
        print(timer.get_time())
    """

    def __init__(self):
        self._start_time = time.time()  # Creation time

    def get_time(self):
        """
        :return: time elapsed since construction in milliseconds
        """
        return (time.time() - self._start_time) * 1000
