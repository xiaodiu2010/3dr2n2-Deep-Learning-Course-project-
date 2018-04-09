import json
from collections import namedtuple
import os
import munch

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    #config = namedtuple("config", config_dict.keys())(*config_dict.values())
    config = munch.DefaultMunch.fromDict(config_dict)
    return config, config_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    return config