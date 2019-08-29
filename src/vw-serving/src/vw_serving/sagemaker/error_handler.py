"""
Module which contains the necessary components to handle MXNet errors and resurface those back to customers
with user friendly messaging. Algorithm authors have the ability to customize the error messaging per algorithm
based on the errors thrown by MXNet in different scenarios.
"""
import logging
import sys

from vw_serving.sagemaker import integration as integ
from vw_serving.sagemaker.exceptions import CustomerError, AlgorithmError
from vw_serving.sagemaker.utils import Preconditions


class BaseErrorMessageMap(object):
    """
    Mapper class to map MXNet error messages to customer friendly error messages for 1P algorithms.
    Individual algorithms can override the :func:`~get_error_message_map` to have customized error messages
    for different failure cases for MXNet.
    """

    def get_error_message_map(self):
        """
        :return the default mapping of MXNet error string patterns to customer friendly messages.
        :rtype Dict
        """
        _default_error_message_map = {
            'out of memory': 'Out of Memory. Please use a larger instance type or reduce batch size. For additional '
                             'hyper-parameters that may have an effect on memory usage, please check the algorithm '
                             'documentation.'
        }
        return _default_error_message_map


def map_mxnet_error_to_sdk_exception(error_msg_map, error_msg):
    """
    Checks if the error from MXNet contains any of the pre-defined error messages and returns CustomerError with the
    user facing message wrapped or AlgorithmError with the original MXNet error if the mapping is not found.
    :param error_msg_map: Mapping of MXNet error string patterns to user facing error messages.
    :type error_msg_map: dict
    :param error_msg: error_msg obtained from MXNet exception stack trace
    :type error_msg: str
    :return CustomerError or AlgorithmError, based on if the mapping is found or not
    :rtype: BaseSdkError
    """
    error_msg = Preconditions.check_not_none(error_msg)
    error_msg_map = Preconditions.check_not_none(error_msg_map)
    for error_str, error_customer_msg in error_msg_map.items():
        if error_str in error_msg:
            return CustomerError(error_customer_msg)
    return AlgorithmError(error_msg)


def _write_public_error_from_sdk_error(sdk_error):
    """Write public error message to the public stream (stdout)"""
    logging.error(sdk_error.public_failure_message(), exc_info=None)


def _write_failure_reason_from_sdk_error(sdk_error):
    """Write failure reason string to the failure file"""
    integ.write_failure_reason(sdk_error.public_failure_message())


def _write_trusted_error_from_sdk_error(sdk_error):
    """Write error details to the trusted error channel"""
    integ.write_trusted_log_exception(sdk_error.private_failure_message())


def report_train_sdk_error_and_terminate(sdk_error):
    """Report SDK error occurred during training and terminate.

    Write public error message to the public stream (stdout).
    Write private error message to the trusted channel.
    Write failure reason to the failure file.

    :param sdk_error: BaseSdkError instance
    """
    _write_public_error_from_sdk_error(sdk_error)  # write to stdout
    _write_failure_reason_from_sdk_error(sdk_error)  # write to /opt/ml/output/failure
    _write_trusted_error_from_sdk_error(sdk_error)  # write to trusted error log

    integ.write_trusted_log_info("worker closed")
    sys.exit(sdk_error.exit_code)


def report_online_inference_sdk_error(sdk_error):
    """Report SDK error occurred during batch inference or online inference.

    Write private error message along with stack trace to the public stream (stdout).

    :param sdk_error: BaseSdkError instance
    """
    logging.exception(sdk_error.private_failure_message())


def report_batch_inference_sdk_error(sdk_error):
    """Report SDK error occurred during batch inference or online inference.

    Write public error message to the public stream (stdout).
    Write private error message to the trusted channel.

    :param sdk_error: BaseSdkError instance
    """
    _write_public_error_from_sdk_error(sdk_error)  # write to stdout
    _write_trusted_error_from_sdk_error(sdk_error)  # write to trusted error log
