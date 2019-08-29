import importlib
import random
import string


def dynamic_import(path):
    """Dynamically import a class from a python module.
    
    Arguments:
        path (str) -- "module.submodule.MyClass"
    
    Returns:
        The corresponding class object if the import was successful.

    Raises:
        ModuleNotFoundError: If the module was not found.
        AttributeError: If the corresponding class was not found in the module.
    """
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def parse_s3_url(s3_url):
    s3_url = s3_url.replace("s3://", "")
    bucket, *key = s3_url.split("/")
    key = "/".join(key)
    return bucket, key


def gen_random_string():
    return ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
