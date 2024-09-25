import os
import re
from configparser import ConfigParser

def load_config():

    config = ConfigParser()
    config.read("config.cfg")
    return config._sections

def set_project_wd():
    cwd = os.getcwd()
    wd = re.match(".*/med_assist", cwd).group(0)

    print(f"Current wd: {cwd}\nChanging to: {wd}")
    os.chdir(wd)

CONFIG = load_config()