from vw_serving.sagemaker.exceptions import CustomerValueError


class DataShapeError(CustomerValueError):
    TEMPLATE = "Incorrect configuration or {channel_name} dataset provided. " \
               "The value of feature_dim hyperparameter '{expected_feature_dim}' doesn't match " \
               "{channel_name} data dimensionality '{data_feature_dim}'. " \
               "Please set feature_dim to the number of features in {channel_name} dataset."

    def __init__(self, channel_name, data_feature_dim, expected_feature_dim):
        """
        :param channel_name: data channel name
        :param data_feature_dim: observed feature dimensionality
        :param expected_feature_dim: expected feature dimensionality provided in training parameters
        """
        super(DataShapeError, self).__init__(self.TEMPLATE.format(
            channel_name=channel_name,
            data_feature_dim=data_feature_dim,
            expected_feature_dim=expected_feature_dim
        ))


class LabelShapeError(CustomerValueError):
    TEMPLATE = "Labeled {channel_name} data required with label size equal 1. " \
               "Label size on {channel_name} dataset found to be {label_size}. " \
               "Please provide a properly labeled labeled {channel_name} dataset."

    def __init__(self, channel_name, label_size):
        """
        :param channel_name: data channel name
        :param label_size: observed label size
        """
        super(LabelShapeError, self).__init__(self.TEMPLATE.format(channel_name=channel_name, label_size=label_size))


def validate_iterator_data_shape(channel_name, data_iter, feature_dim):
    """Validate shape of features.

    Raise a DataShapeError if data shape doesn't match feature_dim.

    :param channel_name: (str) name of data channel
    :param data_iter: (mxnet.io.DataIter) data iterator
    :param feature_dim: (int) expected feature dimensionality
    """

    data_batch_description = data_iter.provide_data
    data_description = data_batch_description[0]
    data_feature_dim = data_description.shape[1]

    if feature_dim != data_feature_dim:
        raise DataShapeError(
            channel_name=channel_name,
            data_feature_dim=data_feature_dim,
            expected_feature_dim=feature_dim
        )


def validate_iterator_label_shape(channel_name, data_iter):
    """Validate label shape.

    Raise a CustomerValueError if data is not labeled or label shape is not (batch_size,) or (batch_size, 1).

    :param channel_name: (str) name of data channel
    :param data_iter: (mxnet.io.DataIter) data iterator
    """

    label_batch_description = data_iter.provide_label
    label_description = label_batch_description[0]
    label_shape = label_description.shape

    if len(label_shape) == 1:  # label is a vector
        label_size = 1
    elif len(label_shape) == 2:  # label is a matrix
        label_size = label_shape[1]
    elif len(label_shape) > 2:  # label is a high-rank tensor
        # Comparison of label_size list to 1 (see below) is going to fail, that is intended behavior.
        # label_size list is then passed to LabelShapeError to be reported to user.
        label_size = label_shape[1:]
    else:
        label_size = 0  # unlabeled data

    if label_size != 1:
        raise LabelShapeError(channel_name=channel_name, label_size=label_size)
