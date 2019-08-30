import json
import warnings

from vw_serving.sagemaker.exceptions import CustomerKeyError, CustomerValueError

LOG_LEVEL = '_log_level'


class ConfigHelper(object):
    """
    Helper class that provides functions to extract configuration values of a particular type.
    """
    AUTO_STRING = "auto"
    STRING_TO_BOOLEAN_DICT = {
        'true': True,
        'false': False
    }

    @classmethod
    def get_int(cls, params, key):
        """
        Retrieves an integer value from configuration dictionary.
        Value must either be an integer or be convertible to an integer.
        Raises a CustomerKeyError if key is not present in the params dictionary.
        Raises a CustomerValueError if provided value is not an int or cannot be converted to an int.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :return: integer value
        """
        try:
            raw_value = params[key]
        except KeyError:
            raise CustomerKeyError(cls._format_missing_hyperparameter_error(key))

        try:
            return int(raw_value)
        except ValueError:
            raise CustomerValueError(cls._format_wrong_hyperparameter_type_error(key, raw_value, "integer"))

    @classmethod
    def get_float(cls, params, key, default=None):
        """
        Retrieves a float value from configuration dictionary.
        Value must either be a float or be convertible to a float.
        Raises a CustomerKeyError if key is not present in the params dictionary.
        Raises a CustomerValueError if provided value is not a float or cannot be converted to a float.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :param default: default value to use if params[key] is missing
        :return: integer value
        """
        if default is None:
            try:
                raw_value = params[key]
            except KeyError:
                raise CustomerKeyError(cls._format_missing_hyperparameter_error(key))
        else:
            warnings.warn("default parameter is deprecated", DeprecationWarning)
            raw_value = params.get(key, default)

        try:
            return float(raw_value)
        except ValueError:
            raise CustomerValueError(cls._format_wrong_hyperparameter_type_error(key, raw_value, "float"))

    @classmethod
    def get_float_or_none(cls, params, key):
        """
        Retrieves a float value from configuration dictionary.
        Value must either be a float or be convertible to a float.
        Returns None if key is not present in the params dictionary.
        Raises a CustomerValueError if provided value is not a float or cannot be converted to a float.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :return: integer value
        """
        try:
            return cls.get_float(params, key)
        except KeyError:
            return None

    @classmethod
    def get_bool(cls, params, key):
        """
        Retrieves a boolean value from configuration dictionary.
        Value must either be a boolean or be convertible to a float.
        Raises a CustomerKeyError if provided value is not a boolean or cannot be converted to a boolean.
        Raises a CustomerValueError if provided value doesn't exist

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :return: integer value
        """
        try:
            raw_value = params[key]
        except KeyError:
            raise CustomerKeyError(cls._format_missing_hyperparameter_error(key))

        if type(raw_value) == bool:
            return raw_value

        try:
            return cls.STRING_TO_BOOLEAN_DICT[raw_value.lower()]
        except (KeyError, AttributeError):
            raise CustomerValueError(cls._format_wrong_hyperparameter_type_error(key, raw_value, "boolean"))

    @classmethod
    def get_bool_or_default(cls, params, key, default):
        """
        Retrieves a boolean value from configuration dictionary.
        Value must either be a boolean or be convertible to a float.
        Raises a CustomerValueError if provided value is not a boolean or cannot be converted to a boolean.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :param default: default value to use if params[key] is missing
        :return: integer value
        """
        warnings.warn("deprecated", DeprecationWarning)
        try:
            return cls.get_bool(params, key)
        except KeyError:
            return default

    @classmethod
    def get_int_default(cls, params, key, default):
        """
        Retrieves an integer value from configuration dictionary.
        Value must either be an integer or be convertible to an integer.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :param default: default value to use if params[key] is missing
        :return: integer value
        """
        warnings.warn("deprecated", DeprecationWarning)
        try:
            return cls.get_int(params, key)
        except KeyError:
            return default

    @classmethod
    def get_int_or_none(cls, params, key):
        """
        Retrieves an integer value from configuration dictionary.
        Value must either be an integer or be convertible to an integer.
        Returns None if value is not provided.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :return: integer value
        """
        try:
            return cls.get_int(params, key)
        except KeyError:
            return None

    @classmethod
    def is_auto_or_absent(cls, params, key):
        """
        Retrieves a string value from configuration dictionary.
        Returns True if the string matches AUTO_STRING or the key is absent.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :return: True if the input string matches AUTO_STRING or the entry is not present
                 False, otherwise
        """
        warnings.warn("deprecated", DeprecationWarning)
        try:
            raw_value = params[key]
            if raw_value == ConfigHelper.AUTO_STRING:
                return True
        except KeyError:
            return True
        return False

    @classmethod
    def is_auto(cls, params, key):
        """
        Retrieves a string value from configuration dictionary.
        Returns True if the string matches AUTO_STRING.

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :return: True if the input string matches AUTO_STRING.
        """
        try:
            value = params[key]
        except KeyError:
            raise CustomerKeyError(cls._format_missing_hyperparameter_error(key))

        return value == ConfigHelper.AUTO_STRING

    @classmethod
    def get_list(cls, params, key, allowed_list=None):
        """
        Retrieves a list from configuration dictionary and validates if allowed_list is not empty or None.
        Value must either be list or string that is convertible to a list.
        Ex: string: '["xyz", "pqr"]' or ["a", "b"]
        Raises a CustomerKeyError if key is not present in the params dictionary.
        Raises a CustomerValueError if provided value is not a list or cannot be converted to list or
        the values in the list are not in the non-empty allowed_list

        :param params: configuration dictionary
        :param key: key to use for value retrieval
        :param allowed_list: list of allowed values, No validation is performed if allowed_list is empty or None.
        :return: list object

        """
        try:
            raw_value = params[key]
        except KeyError:
            raise CustomerKeyError(cls._format_missing_hyperparameter_error(key))

        if type(raw_value) is list:
            list_val = raw_value
        else:
            try:
                list_val = json.loads(raw_value)
            except ValueError as e:
                raise CustomerValueError("Hyperparameter must be valid json, but found {}:".format(key), e)

            if type(list_val) is not list:
                raise CustomerValueError("Expected list type for hyperparameter: {}, "
                                         "found value: {} of type: {}".format(key, list_val, type(list_val)))
        if allowed_list:
            cls._validate_list(list_val, allowed_list, key)
        return list_val

    @classmethod
    def _validate_list(cls, input_list, allowed_list, key):
        """
        Checks if the item in input_list is in the allowed_list
        :param input_list: list of items to validate
        :param allowed_list: list of allowed values
        :return:
        """
        invalid_input = set(input_list) - set(allowed_list)
        if invalid_input:
            raise CustomerValueError(
                "The list of values for hyperparameter '{key}' contains invalid value: {invalid_input}. The supported "
                "values are: {allowed_list}. Please provide supported values for hyperparamter '{key}' and try again."
                .format(key=key, invalid_input=list(invalid_input), allowed_list=allowed_list))

    @classmethod
    def _format_missing_hyperparameter_error(cls, key):
        return "Hyperparameter '{key}' is missing. Please provide a value for hyperparameter '{key}' in the request " \
               "and try again.".format(key=key)

    @classmethod
    def _format_wrong_hyperparameter_type_error(cls, key, value, expected_type):
        return "The value for hyperparameter '{key}' should be of type {expected_type}, but provided '{value}'. " \
               "Please provide a valid '{expected_type}' value and try again.".format(key=key, value=value,
                                                                                      expected_type=expected_type)
