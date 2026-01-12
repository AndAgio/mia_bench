import os
import pathlib
import yaml


PATH_REPO = pathlib.Path(__file__).parent.parent.parent


def load_secrets_yaml():
    with open(os.path.join(PATH_REPO, "secrets.yml"), "r") as readfile:
        secrets = yaml.safe_load(readfile)
    return secrets
