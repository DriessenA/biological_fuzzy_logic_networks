import json

base_config_path = "/u/adr/Code/biological_fuzzy_logic_networks/biological_fuzzy_logic_networks/Synthetic_experiments/base_measured_nodes.json"
config_dir = "/dccstor/ipc1/CAR/BFN/Model/MeasuredNodes/Configs/"
max_nodes = 13

with open(base_config_path) as f:
    base_config = json.load(f)
f.close()

base_path = base_config["out_dir"]

for i in range(max_nodes):
    config = base_config.copy()
    config["n_nodes_measured"] = i + 1

    config["out_dir"] = f"{base_path}{i+1}_nodes/"

    with open(f"{config_dir}{i+1}_nodes_config.json", "w") as file:
        json.dump(config, file)
    file.close()
