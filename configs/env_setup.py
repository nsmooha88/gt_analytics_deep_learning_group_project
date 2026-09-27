import yaml
from pathlib import Path

class LocalVariables:
    def __init__(self, yaml_config_path="../local_config.yaml"):
        # NOTE: might need to change the config path since it depends on the path of which python file you run
        '''
        Set up variables using your local system paths and hyperparameters
        Custom values should be set in local_config.yaml
        Example use to get path to source image path for segmentation model:
        lv = LocalVariables()
        lv.souce_seg

        :param yaml_config_path: path to the yaml config file, default is local_config.yaml
        '''

        # load everything from local_config.yaml
        with open(yaml_config_path, 'r') as file:
            config = yaml.safe_load(file)

        # file paths
        root = Path(config["files"]["root"])
        self.source_seg = root / config["files"]["source_seg"]
        self.target_seg = root / config["files"]["target_seg"]
        self.source_class = root / config["files"]["source_class"]
        self.target_class = root / config["files"]["target_class"]

        self.model = config["train"]["model"]
        self.mode = config["train"]["mode"]

        # dictionaries containing hyperparameters
        self.seg_params = config["seg_config"]
        self.class_params = config["class_config"]
        self.seg_tuning_params = config["seg_tuning"]

