import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

from biological_fuzzy_logic_networks.utils import read_sif
from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import DREAMBioFuzzNet
from biological_fuzzy_logic_networks.label_shuffle import create_shuffled_subclass


def format_data_dicts(
    data_df: pd.DataFrame, input_nodes: List[str], value_nodes: List[str], prefix=""
):
    input_df = data_df[input_nodes]
    value_df = data_df[value_nodes]
    input_dict = {
        k: torch.tensor(v.values) for k, v in input_df.to_dict(orient="series").items()
    }
    value_dict = {
        k: torch.tensor(v.values) for k, v in value_df.to_dict(orient="series").items()
    }
    inhibitor_dict = {k: torch.ones_like(v) for k, v in value_dict.items()}
    return {
        f"{prefix}input": input_dict,
        f"{prefix}ground_truth": value_dict,
        f'{prefix or "train_"}inhibitors': inhibitor_dict,
    }


def prepare_data(data_df: pd.DataFrame, input_nodes: List[str], value_nodes: List[str]):
    train_df, test_df = train_test_split(data_df, train_size=0.9)
    train_df, valid_df = train_test_split(train_df, train_size=0.8)
    return dict(
        **format_data_dicts(train_df, input_nodes, value_nodes, prefix=""),
        **format_data_dicts(valid_df, input_nodes, value_nodes, prefix="valid_"),
        **format_data_dicts(test_df, input_nodes, value_nodes, prefix="test_"),
    )


def train_model_and_get_predictions(
    nodes,
    edges,
    optim_data: dict,
    data_dicts: dict,
    shuffling: bool,
    nrep: int = 10,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.01,
):
    all_predictions = []
    for i in range(nrep):
        if shuffling:
            bfn = create_shuffled_subclass(DREAMBioFuzzNet)(nodes=nodes, edges=edges)
        else:
            bfn = DREAMBioFuzzNet(nodes=nodes, edges=edges)
        _, curr_best_val_loss, _ = bfn.conduct_optimisation(
            epochs=10, batch_size=64, learning_rate=0.01, **optim_data
        )
        print(["Shuffled" if shuffling else "Unshuffled"], i, curr_best_val_loss)
        test_gt = data_dicts["test_ground_truth"]
        test_inh = data_dicts["test_inhibitors"]
        n_test = len(test_gt["mek12"])

        with torch.no_grad():
            bfn.initialise_random_truth_and_output(n_test)
            bfn.set_network_ground_truth(test_gt)
            bfn.sequential_update(input_nodes=bfn.root_nodes, inhibition=test_inh)
        data = {}
        for n in bfn.biological_nodes:
            if n not in bfn.root_nodes:
                data[f"{n}_pred"] = bfn.nodes[n]["output_state"].numpy()
            data[n] = bfn.nodes[n]["ground_truth"].numpy()
        test_predictions = pd.DataFrame(data)
        all_predictions.append(test_predictions)

    return pd.concat(all_predictions)


def node_shuffling_exp(pkn_path, data_path, nrep):
    nodes, edges = read_sif(pkn_path)
    df = pd.read_csv(f"{data_path}teacher_data.csv", index_col=0)
    bfn = DREAMBioFuzzNet(nodes=nodes, edges=edges)
    input_nodes = bfn.root_nodes
    value_nodes = bfn.biological_nodes
    data_dicts = prepare_data(df, input_nodes, value_nodes)
    optim_data = {k: v for k, v in data_dicts.items() if "test" not in k}

    network_pred = train_model_and_get_predictions(
        nodes=nodes,
        edges=edges,
        optim_data=optim_data,
        shuffling=False,
        data_dicts=data_dicts,
        nrep=nrep,
    )

    shuffled_network_pred = train_model_and_get_predictions(
        nodes=nodes,
        edges=edges,
        optim_data=optim_data,
        shuffling=True,
        data_dicts=data_dicts,
        nrep=nrep,
    )

    network_pred.to_csv(f"{data_path}network_pred.csv")
    shuffled_network_pred.to_csv(f"{data_path}shuffled_network_pred.csv")


if __name__ == "__main__":
    data_path = "/dccstor/ipc1/CAR/BFN/Model/Shuffling/"
    pkn_path = "/dccstor/ipc1/CAR/BFN/LiverDREAM_PKN.sif"
    nrep = 10

    node_shuffling_exp(pkn_path=pkn_path, data_path=data_path, nrep=nrep)
