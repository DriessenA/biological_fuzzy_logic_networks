import json
from sklearn.model_selection import ParameterGrid

param_dict = {
    "inhibited_node": [
        "mek12",
        "erk12",
        "mkk4",
        "jnk12",
        "ikk",
        "ikb",
        "ras",
        "map3k7",
        "igf1",
        "pi3k",
        "il1a",
        "map3k1",
        "tgfa",
        "tnfa",
        "akt",
        "p38",
        "hsp27",
    ],
    "k_inhibition": [-1e2, -10, -5, -1, -0.5],
    "divide_inhibition": [10, 20, 30, 40, 50],
}

base_config_path = "/u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/base_perturb_config.json"
config_dir = "/dccstor/ipc1/CAR/BFN/Model/Perturbation_Liver/Configs/"
n_repeats = 5

with open(base_config_path) as f:
    base_config = json.load(f)
f.close()

base_path = base_config["out_dir"]

for i, params in enumerate(ParameterGrid(param_dict)):
    config = base_config.copy()
    config["param_setting"] = i

    for k, v in params.items():
        config[k] = v

    for n in range(n_repeats):
        config["repeat"] = n
        config["out_dir"] = f"{base_path}param_{i}_repeat_{n}"

        with open(f"{config_dir}param_{i}_repeat_{n}_config.json", "w") as file:
            json.dump(config, file)
        file.close()
