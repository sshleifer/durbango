from transformers.configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP



from transformers.hf_api import HfApi

import os
import json

import sys


def update_config(model_identifier, updates):
    api = HfApi()
    model_list = api.model_list()
    model_dict = [
        model_dict
        for model_dict in model_list
        if model_dict.modelId == model_identifier
    ][0]

    model_identifier = "_".join(model_identifier.split("/"))

    http = "https://s3.amazonaws.com/"
    hf_url = "models.huggingface.co/"
    config_path_aws = http + hf_url + model_dict.key
    file_name = f"./{model_identifier}_config.json"

    bash_command = f"curl {config_path_aws} > {file_name}"
    os.system(bash_command)

    with open(file_name) as f:
        config_json = json.load(f)

    bash_command = "rm {}".format(file_name)
    os.system(bash_command)

    ##### HERE YOU SHOULD STATE WHICH PARAMS WILL BE CHANGED #####
    config_json.update(updates)

    # save config as it was saved before
    with open(file_name, "w") as f:
        json.dump(config_json, f, indent=2, sort_keys=True)

    # upload new config
    bash_command = f"s3cmd cp {file_name} s3://{hf_url + model_dict.key}"
    os.system(bash_command)
