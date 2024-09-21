from biological_fuzzy_logic_networks.DREAM_analysis.utils import (
    prepare_cell_line_data,
    cl_data_to_input,
)
from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import DREAMBioFuzzNet
import pandas as pd
from typing import List, Union, Sequence
from sklearn.metrics import r2_score
import mlflow
import click
import json
import torch
import pickle as pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    data_file: Union[List, str],
    output_dir: str,
    time_point: int = 9,
    non_marker_cols: Sequence[str] = (
        "treatment",
        "cell_line",
        "time",
        "cellID",
        "fileID",
    ),
    treatment_col_name: str = "treatment",
    sample_n_cells: Union[int, bool] = False,
    filter_starved_stim: bool = True,
    sel_condition: str = None,
    scaler_type: str = "minmax",
    add_root_values: bool = True,
    input_value: float = 1,
    root_nodes: Sequence[str] = ("EGF", "SERUM"),
    replace_zero_inputs: Union[bool, float] = False,
    train_treatments: List[str] = None,
    valid_treatments: List[str] = None,
    train_cell_lines: List[str] = None,
    valid_cell_lines: List[str] = None,
    test_cell_lines: List[str] = None,
    inhibition_value: Union[int, float] = 1.0,
    learning_rate: float = 1e-3,
    n_epochs: int = 20,
    batch_size: int = 300,
    checkpoint_path: str = None,
    convergence_check: bool = False,
    shuffle_nodes: bool = False,
    patience: int = 20,
    **extras,
):
    model = DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(pkn_sif)
    cl_data = prepare_cell_line_data(
        data_file=data_file,
        time_point=time_point,
        non_marker_cols=non_marker_cols,
        treatment_col_name=treatment_col_name,
        filter_starved_stim=filter_starved_stim,
        sample_n_cells=sample_n_cells,
        sel_condition=sel_condition,
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
        scale_type=scaler_type,
        add_root_values=add_root_values,
        input_value=input_value,
        root_nodes=root_nodes,
        replace_zero_inputs=replace_zero_inputs,
        balance_data=True,
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
        logger=mlflow,
        patience=patience,
    )

    print("loss: ", loss)
    print("best loss: ", best_val_loss)

    with open(f"{output_dir}scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    loss.to_csv(f"{output_dir}loss.csv")
    train.to_csv(f"{output_dir}train_data.csv")
    valid.to_csv(f"{output_dir}valid_data.csv")

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
    valid_inhibitors = {k: v.to(device) for k, v in valid_inhibitors.items()}
    ckpt = torch.load(f"{checkpoint_path}/model.pt", map_location=torch.device(device))
    model = DREAMBioFuzzNet.build_DREAMBioFuzzNet_from_file(pkn_sif)
    model.load_from_checkpoint(ckpt["model_state_dict"])

    with torch.no_grad():
        model.initialise_random_truth_and_output(len(valid))
        model.set_network_ground_truth(valid_data)
        print(model.output_states)
        model.sequential_update(model.root_nodes, valid_inhibitors)
        val_output_states = pd.DataFrame(
            {k: v.cpu().numpy() for k, v in model.output_states.items()}
        )

    # Vaidation performance
    node_r2_scores = {}
    for node in valid_data.keys():
        node_r2_scores[f"val_r2_{node}"] = r2_score(
            valid[node], val_output_states[node]
        )
    # mlflow.log_metric("best_val_loss", best_val_loss)
    # mlflow.log_metrics(node_r2_scores)

    # Save outputs
    train_inhibitors = {k: v.to("cpu") for k, v in train_inhibitors.items()}
    valid_inhibitors = {k: v.to("cpu") for k, v in valid_inhibitors.items()}
    pd.DataFrame(train_inhibitors).to_csv(f"{output_dir}train_inhibitors.csv")
    pd.DataFrame(valid_inhibitors).to_csv(f"{output_dir}valid_inhibitors.csv")
    val_output_states.to_csv(f"{output_dir}valid_output_states.csv")


@click.command()
@click.argument("config_path")
def main(config_path):
    with open(config_path) as f:
        config = json.load(f)
    f.close()

    # with mlflow_tunnel(host="mlflow") as tunnel:
    #     remote_port = tunnel[5000]
    #     mlflow.set_tracking_uri(f"http://localhost:{remote_port}")
    #     mlflow.set_experiment(config["experiment_name"])

    #     job_id = get_environ_var("LSB_JOBID", fail_gracefully=True)
    #     mlflow.log_param("ccc_job_id", job_id)
    #     log_params = {
    #         x: [y.split("/")[-1] for y in config[x]]
    #         if x
    #         in [
    #             "valid_cell_lines",
    #             "test_cell_lines",
    #             "train_cell_lines",
    #         ]
    #         and not config[x] is None
    #         else config[x]
    #         for x in config
    #     }
    #     log_params = {x: y for x, y in log_params.items() if len(str(y)) < 500}
    #     mlflow.log_params(log_params)
    train_network(**config)


if __name__ == "__main__":
    main()
