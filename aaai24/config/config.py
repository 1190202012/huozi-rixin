import json
import os
import time

import dynamic_yaml
import numpy as np
import torch
import yaml
from loguru import logger
from tqdm import tqdm

class Config:
    """Configurator module that load the defined parameters."""

    def __init__(self, config_file, debug=False):
        """

        Load parameters and set log level.

        Args:
            config_file (str): path to the config file, which should be in ``yaml`` format.
                You can use default config provided in the `Github repo`_, or write it by yourself.
            debug (bool, optional): whether to enable debug function during running. Defaults to False.

        """
        self.opt = self.load_yaml_configs(config_file)

        # gpu
        gpu = self.opt["gpu"]
        
        if gpu is None:
            gpu = [i for i in range(torch.cuda.device_count())]
        elif type(gpu) is int:
            gpu = [gpu]
        elif type(gpu) is str:
            gpu = [int(i) for i in range(len(gpu.replace(" ", "").split(',')))]
        
        self.opt["gpu"] = gpu
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(gpu)
        
        if self.opt["mode"] == "demo":
            torch.multiprocessing.set_start_method('spawn')
        elif self.opt["mode"] == "eval":
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"

        self.opt["time"] = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        
        # check args
        self.check_config(self.opt)

        # seed
        self.set_seed(self.opt["seed"], len(self.opt["gpu"]))
                
        # log
        log_name = self.opt.get("log_name", self.opt["time"]) + ".log"
        if not os.path.exists("log"):
            os.makedirs("log")
        logger.remove()
        if debug:
            level = 'DEBUG'
        else:
            level = 'INFO'
        logger.add(os.path.join("log", log_name), level=level)
        logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level=level)

        logger.info("[Config]" + '\n' + json.dumps(self.opt, indent=4))

    @staticmethod
    def load_yaml_configs(filename):
        """This function reads ``yaml`` file to build config dictionary

        Args:
            filename (str): path to ``yaml`` config

        Returns:
            dict: config

        """
        config_dict = dict()
        with open(filename, 'r', encoding='utf-8') as f:
            config_dict.update(yaml.safe_load(dynamic_yaml.dump(dynamic_yaml.load(f.read()))))
        return config_dict

    @staticmethod
    def set_seed(seed: int, gpu_num: int):
        """
        Set seed for numpy and torch
        :param seed: int
        :param gpu_num: num of gpu to use
        :return: None
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if gpu_num > 0:
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def check_config(config):
        if config["test_file"] is not None:
            assert os.path.isfile(config["test_file"]), f"{config['test_file']} doesn't exists."
        
        # makedir for output_dir
        if config["output_dir"] is not None:
            config["output_dir"] = config["output_dir"] + config["time"] + "/"
            
            if config["ddp"]:
                config["output_dir"] = config["output_dir"] + "$rank$/"
            
            if "$rank$" in config["output_dir"]:
                for rank in range(len(config["gpu"])):
                    _dir = config["output_dir"].replace("$rank$", str(rank))
                    if not os.path.exists(_dir):
                        os.makedirs(_dir, exist_ok=False)
            else:     
                if not os.path.exists(config["output_dir"]):
                    os.makedirs(config["output_dir"], exist_ok=False)   

        # get output_path
        if config["output_dir"] is not None and config["result_file"] is not None:
            config["result_path"] = config["output_dir"] + config["result_file"]
        else:
            config["result_path"] = None
        
        if config["output_dir"] is not None and config["result_info_file"] is not None:
            config["result_info_path"] = config["output_dir"] + config["result_info_file"]
        else:
            config["result_info_path"] = None
        
        for name, module in config["ModuleConfig"].items():
            if module["ban"]:
                continue
            
            if config["cache_dir"] is not None and module["load_result_file"] is not None:
                module["load_result_path"] = config["cache_dir"] + module["load_result_file"]
                assert os.path.isfile(module["load_result_path"]), f"[{name}]: load result path doesn't exists."
            else:
                module["load_result_path"] = None
            
            if config["cache_dir"] is not None and module["load_info_file"] is not None:
                module["load_info_path"] = config["cache_dir"] + module["load_info_file"]
                assert os.path.isfile(module["load_info_path"]), f"[{name}]: load info path doesn't exists."
            else:
                module["load_info_path"] = None
            
            if config["output_dir"] is not None and module["save_result_file"] is not None:
                module["save_result_path"] = config["output_dir"] +  module["save_result_file"]
            else:
                module["save_result_path"] = None
            
            if config["output_dir"] is not None and module["save_info_file"] is not None:                             
                module["save_info_path"] = config["output_dir"] +  module["save_info_file"]
            else:
                module["save_info_path"] = None
                            
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.opt[key] = value

    def __getitem__(self, item):
        if item in self.opt:
            return self.opt[item]
        else:
            return None

    def get(self, item, default=None):
        """Get value of corresponding item in config

        Args:
            item (str): key to query in config
            default (optional): default value for item if not found in config. Defaults to None.

        Returns:
            value of corresponding item in config

        """
        if item in self.opt:
            return self.opt[item]
        else:
            return default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.opt

    def __str__(self):
        return str(self.opt)

    def __repr__(self):
        return self.__str__()
