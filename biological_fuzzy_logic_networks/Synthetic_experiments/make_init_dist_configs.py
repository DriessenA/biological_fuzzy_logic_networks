import json
from sklearn.model_selection import ParameterGrid

param_dict = {"dist_name": ["normal", "bimodal", "uniform"]}

base_config_path = "/u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/base_init_dist_config.json"
config_dir = "/dccstor/ipc1/CAR/BFN/Model/init_dists/Configs/"
n_repeats = 5

with open(base_config_path) as f:
    base_config = json.load(f)
f.close()

base_path = base_config["out_dir"]

for params in ParameterGrid(param_dict):
    config = base_config.copy()

    dist_name = params["dist_name"]

    for k, v in params.items():
        config[k] = v

    for n in range(n_repeats):
        config["repeat"] = n
        config["out_dir"] = f"{base_path}{dist_name}_repeat_{n}"

        with open(f"{config_dir}{dist_name}_repeat_{n}_config.json", "w") as file:
            json.dump(config, file)
        file.close()
