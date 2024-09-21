from biological_fuzzy_logic_networks.DREAM_analysis.utils import (
    create_bfz,
    prepare_cell_line_data,
    cl_data_to_input,
)
import pandas as pd
import numpy as np
from typing import List, Union
from sklearn.metrics import r2_score
import click
import json
import torch
import pickle as pickle
import os


def get_environ_var(env_var_name, fail_gracefully=True):
    try:
        assert (
            env_var_name in os.environ
        ), f"Environment variable ${env_var_name} not set, are you on a CCC job?"
        var = os.environ[env_var_name]
    except AssertionError:
        if not fail_gracefully:
            raise
        else:
            var = None

    return var


def train_network(
    pkn_sif: str,
    network_class: str,
    data_file: List,
    output_dir: str,
    time_point: int = 9,
    non_marker_cols: List[str] = ["treatment", "cell_line", "time", "cellID", "fileID"],
    treatment_col_name: str = "treatment",
    sample_n_cells: Union[int, bool] = False,
    filter_starved_stim: bool = True,
    minmaxscale: bool = True,
    add_root_values: bool = True,
    input_value: float = 1,
    root_nodes: List[str] = ["EGF", "SERUM"],
    replace_zero_inputs: Union[bool, float] = False,
    train_treatments: List[str] = None,
    valid_treatments: List[str] = None,
    test_treatments: List[str] = None,
    train_cell_lines: List[str] = None,
    valid_cell_lines: List[str] = None,
    test_cell_lines: List[str] = None,
    inhibition_value: Union[int, float] = 1.0,
    learning_rate: float = 1e-3,
    n_epochs: int = 20,
    batch_size: int = 300,
    checkpoint_path: str = None,
    convergence_check: bool = False,
    **extras,
):
    model = create_bfz(pkn_sif, network_class)
    cl_data = prepare_cell_line_data(
        data_file=data_file,
        time_point=time_point,
        non_marker_cols=non_marker_cols,
        treatment_col_name=treatment_col_name,
        filter_starved_stim=filter_starved_stim,
        sample_n_cells=sample_n_cells,
    )

    # Load train and valid data
    (
        train_data,
        valid_data,
        train_inhibitors,
        valid_inhibitors,
        train_input,
        valid_input,
        train,
        valid,
        scaler,
    ) = cl_data_to_input(
        data=cl_data,
        model=model,
        train_treatments=train_treatments,
        valid_treatments=valid_treatments,
        train_cell_lines=train_cell_lines,
        valid_cell_lines=valid_cell_lines,
        inhibition_value=inhibition_value,
        minmaxscale=minmaxscale,
        add_root_values=add_root_values,
        input_value=input_value,
        root_nodes=root_nodes,
        replace_zero_inputs=replace_zero_inputs,
        balance_data=True,
    )

    # Load test data
    # Test set performance
    cl_data = prepare_cell_line_data(
        data_file=data_file,
        time_point=time_point,
        non_marker_cols=non_marker_cols,
        treatment_col_name=treatment_col_name,
        sample_n_cells=False,
        filter_starved_stim=filter_starved_stim,
    )

    (
        _,
        test_data,
        _,
        test_inhibitors,
        _,
        test_input,
        _,
        test,
        scaler,
    ) = cl_data_to_input(
        data=cl_data,
        model=model,
        train_treatments=train_treatments,
        valid_treatments=test_treatments,
        train_cell_lines=train_cell_lines,
        valid_cell_lines=valid_cell_lines,
        inhibition_value=inhibition_value,
        minmaxscale=scaler,
        add_root_values=add_root_values,
        input_value=input_value,
        root_nodes=root_nodes,
        do_split=False,
        replace_zero_inputs=replace_zero_inputs,
        balance_data=False,
    )

    # Optimize model
    loss, best_val_loss, loop_states = model.conduct_optimisation(
        input=train_input,
        valid_input=valid_input,
        ground_truth=train_data,
        valid_ground_truth=valid_data,
        train_inhibitors=train_inhibitors,
        valid_inhibitors=valid_inhibitors,
        epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        checkpoint_path=checkpoint_path,
        convergence_check=convergence_check,
    )

    if convergence_check:
        temp = {
            idx: {m: v.detach().numpy() for (m, v) in m.items()}
            for (idx, m) in loop_states.items()
        }
        loop_states_to_save = pd.concat(
            [pd.DataFrame(v) for k, v in temp.items()],
            keys=temp.keys(),
            names=["time", ""],
        ).reset_index("time", drop=False)
        loop_states_to_save.to_csv(f"{output_dir}loop_states.csv")

    # Load best model and evaluate:
    ckpt = torch.load(f"{checkpoint_path}/model.pt")
    model = create_bfz(pkn_sif, network_class)
    model.load_from_checkpoint(ckpt["model_state_dict"])
    with torch.no_grad():
        model.initialise_random_truth_and_output(len(valid))
        model.set_network_ground_truth(valid_data)
        model.sequential_update(model.root_nodes, valid_inhibitors)
        val_output_states = pd.DataFrame(
            {k: v.numpy() for k, v in model.output_states.items()}
        )

        model.initialise_random_truth_and_output(len(test))
        model.set_network_ground_truth(test_data)
        model.sequential_update(model.root_nodes, test_inhibitors)
        test_output_states = pd.DataFrame(
            {k: v.numpy() for k, v in model.output_states.items()}
        )

    # Vaidation performance
    node_r2_scores = {}
    for node in valid_data.keys():
        node_r2_scores[f"val_r2_{node}"] = r2_score(
            valid[node], val_output_states[node]
        )

    # Test performance
    node_r2_scores = {}
    node_mse = {}
    for node in test_data.keys():
        node_r2_scores[f"test_r2_{node}"] = r2_score(
            test[node], test_output_states[node]
        )
        node_mse[f"test_mse_{node}"] = sum(
            (np.array(test[node]) - np.array(test_output_states[node])) ** 2
        ) / len(test)


    # Save outputs
    with open(f"{output_dir}scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    val_output_states.to_csv(f"{output_dir}valid_output_states.csv")
    test_output_states.to_csv(f"{output_dir}test_output_states.csv")
    loss.to_csv(f"{output_dir}loss.csv")
    train.to_csv(f"{output_dir}train_data.csv")
    valid.to_csv(f"{output_dir}valid_data.csv")
    test.to_csv(f"{output_dir}test_data.csv")
    pd.DataFrame(train_inhibitors).to_csv(f"{output_dir}train_inhibitors.csv")
    pd.DataFrame(valid_inhibitors).to_csv(f"{output_dir}valid_inhibitors.csv")
    pd.DataFrame(test_inhibitors).to_csv(f"{output_dir}test_inhibitors.csv")


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    train_network(**config)


if __name__ == "__main__":
    main()
