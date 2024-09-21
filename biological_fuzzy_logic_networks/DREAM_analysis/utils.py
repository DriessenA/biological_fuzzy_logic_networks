from typing import List, Union, Type, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from torch import DoubleTensor

from biological_fuzzy_logic_networks.DREAM.DREAMBioFuzzNet import (
    DREAMBioFuzzNet,
    DREAMBioMixNet,
)
from biological_fuzzy_logic_networks.DREAM_analysis.scalers import ClippingScaler
from biological_fuzzy_logic_networks.biofuzznet import BioFuzzNet
from biological_fuzzy_logic_networks.biomixnet import BioMixNet
from biological_fuzzy_logic_networks.label_shuffle import create_shuffled_subclass
from biological_fuzzy_logic_networks.utils import read_sif


def get_scaler(scale_type):
    if scale_type.lower() == "minmax":
        scaler = MinMaxScaler()
    elif scale_type == "quantile":
        scaler = QuantileTransformer()
    elif scale_type == "clippping":
        scaler = ClippingScaler()
    else:
        raise Exception(
            f"Scaler {scale_type} not found use one of minmax, clipping or quantile"
        )

    return scaler


def create_bfz(pkn_sif: str, network_class: str, shuffle_nodes: bool = False):
    nodes, edges = read_sif(pkn_sif)
    class_dispatch: Dict[str, Type[BioFuzzNet]] = {
        "dreambiofuzznet": DREAMBioFuzzNet,
        "dreambiomixnet": DREAMBioMixNet,
        "biomixnet": BioMixNet,
        "biofuzznet": BioFuzzNet,
    }

    try:
        selected_cls = class_dispatch[network_class.lower()]
    except KeyError as e:
        raise Exception(
            "network_class arguement not recognised, should be one of",
            "DREAMBioFuzzNet, DREAMBioMixNet, BioFuzzNet, BioMixNet",
        ) from e

    if shuffle_nodes:
        selected_cls = create_shuffled_subclass(selected_cls)

    model = selected_cls(nodes=nodes, edges=edges)
    return model


def prepare_cell_line_data(
    data_file: Union[List, str],
    time_point: int = 9,
    sel_condition: str = None,
    non_marker_cols: List[str] = ["treatment", "cell_line", "time", "cellID", "fileID"],
    treatment_col_name: str = "treatment",
    sample_n_cells: Union[int, bool] = False,
    filter_starved_stim: bool = True,
    **extras,
):
    print(type(data_file))
    if isinstance(data_file, str):
        cl_data = pd.read_csv(data_file)
    elif isinstance(data_file, List):
        data = []
        for file_name in data_file:
            d = pd.read_csv(file_name)
            data.append(d)
        cl_data = pd.concat(data)
    else:
        raise Exception("`data_file` should be a string or list of strings")

    print(cl_data["cell_line"].unique())
    data_to_nodes_map = data_to_nodes_mapping()
    inhibitor_map = inhibitor_mapping()

    cl_data = cl_data[cl_data["time"] == time_point]
    if sel_condition:
        cl_data = cl_data[cl_data[treatment_col_name] == sel_condition]

    if filter_starved_stim:
        cl_data = cl_data[cl_data["treatment"] != "full"]
        cl_data = cl_data[cl_data["time"] != 0]

    if sample_n_cells:
        replacement = (
            False
            if all(
                cl_data.groupby(["cell_line", "treatment", "time"]).size()
                > sample_n_cells
            )
            else True
        )
        cl_data = cl_data.groupby(["cell_line", "treatment", "time"]).sample(
            n=sample_n_cells, replace=replacement
        )

    cl_data.loc[:, "inhibitor"] = [
        inhibitor_map[treatment] for treatment in cl_data[treatment_col_name]
    ]

    cl_data = cl_data.rename(columns=data_to_nodes_map)

    return cl_data


def split_data(
    data,
    train_treatments,
    valid_treatments,
    train_cell_lines,
    valid_cell_lines,
    do_split: bool = True,
):

    treatment_split = True
    cell_line_split = True
    if train_treatments is None and valid_treatments is None:
        treatment_split = False
    if train_cell_lines is None and valid_cell_lines is None:
        cell_line_split = False

    if not do_split:
        train = data.copy()
        valid = None
    elif not treatment_split and not cell_line_split and do_split:
        train, valid = train_test_split(data)

    else:
        if not treatment_split:
            train_inhibitors = list(data["inhibitor"].unique())
            valid_inhibitors = list(data["inhibitor"].unique())

        else:
            if train_treatments is not None and valid_treatments is not None:
                if not isinstance(train_treatments, List):
                    train_treatments = [train_treatments]
                if not isinstance(valid_treatments, List):
                    valid_treatments = [valid_treatments]

                if (
                    not len(set(train_treatments).intersection(set(valid_treatments)))
                    == 0
                ):
                    raise Exception("Given train and validation treatments overlap")
                else:
                    inhibitor_map = inhibitor_mapping()
                    train_inhibitors = [
                        inhibitor_map[treatment] for treatment in train_treatments
                    ]
                    valid_inhibitors = [
                        inhibitor_map[treatment] for treatment in valid_treatments
                    ]

            elif train_treatments is not None:
                print("Splitting based on train treatment(s)")
                if not isinstance(train_treatments, List):
                    train_treatments = [train_treatments]

                inhibitor_map = inhibitor_mapping()
                train_inhibitors = [
                    inhibitor_map[treatment] for treatment in train_treatments
                ]
                valid_inhibitors = [
                    inhibitor
                    for inhibitor in data["inhibitor"].unique()
                    if inhibitor not in train_inhibitors
                ]

            elif valid_treatments is not None:
                print("Splitting based on validation treatment(s)")
                if not isinstance(valid_treatments, List):
                    valid_treatments = [valid_treatments]

                inhibitor_map = inhibitor_mapping()
                valid_inhibitors = [
                    inhibitor_map[treatment] for treatment in valid_treatments
                ]
                train_inhibitors = [
                    inhibitor
                    for inhibitor in data["inhibitor"].unique()
                    if inhibitor not in valid_inhibitors
                ]

        if not cell_line_split:
            train_cell_lines = list(data["cell_line"].unique())
            valid_cell_lines = list(data["cell_line"].unique())
        else:
            if train_cell_lines is not None and valid_cell_lines is not None:
                if not isinstance(train_cell_lines, List):
                    train_cell_lines = [train_cell_lines]
                if not isinstance(valid_cell_lines, List):
                    valid_cell_lines = [valid_cell_lines]

                if (
                    not len(set(train_cell_lines).intersection(set(valid_cell_lines)))
                    == 0
                ):
                    raise Exception("Given train and validation cell lines overlap")

            elif train_cell_lines is not None:
                print("Splitting based on train cell line(s)")
                if not isinstance(train_cell_lines, List):
                    train_cell_lines = [train_cell_lines]

                valid_cell_lines = [
                    cl
                    for cl in data["cell_line"].unique()
                    if cl not in train_cell_lines
                ]

            elif valid_cell_lines is not None:
                print("Splitting based on validation cell line(s)")
                if not isinstance(valid_cell_lines, List):
                    valid_cell_lines = [valid_cell_lines]

                train_cell_lines = [
                    cl
                    for cl in data["cell_line"].unique()
                    if cl not in valid_cell_lines
                ]

        train = data.loc[
            (data["cell_line"].isin(train_cell_lines))
            & data["inhibitor"].isin(train_inhibitors),
            :,
        ]
        valid = data.loc[
            (data["cell_line"].isin(valid_cell_lines))
            & data["inhibitor"].isin(valid_inhibitors),
            :,
        ]

    return train, valid


def cl_data_to_input(
    data,
    model,
    train_treatments: List[str] = None,
    valid_treatments: List[str] = None,
    train_cell_lines: List[str] = None,
    valid_cell_lines: List[str] = None,
    inhibition_value: Union[int, float] = 1.0,
    scale_type: str = "minmax",
    scaler=None,
    add_root_values: bool = True,
    input_value: float = 1,
    root_nodes: List[str] = ["EGF", "SERUM"],
    do_split: bool = True,
    replace_zero_inputs: Union[bool, float] = False,
    **extras,
):
    markers = [c for c in data.columns if c in model.nodes()]
    if isinstance(replace_zero_inputs, float):
        do_replace = True
        replace_value = replace_zero_inputs
    elif replace_zero_inputs:
        do_replace = True
        replace_value = 1e-9
    else:
        do_replace = False

    data = data.dropna(subset=markers, axis=0)

    train, valid = split_data(
        data,
        train_treatments,
        valid_treatments,
        train_cell_lines,
        valid_cell_lines,
        do_split=do_split,
    )

    if not scaler and not scale_type:
        raise Warning("No scaler type or scaler provided, using unscaled data")
    if scaler:
        if scaler and scale_type:
            raise Warning("Scaler provided, ignoring `scale type`")
        train[markers] = scaler.transform(train[markers])
        t = train[markers]
        t[t < 0] = 1e-9
        t[t > 1] = 1
        train[markers] = t
        if valid is not None:
            valid[markers] = scaler.transform(valid[markers])
            t = valid[markers]
            t[t < 0] = 1e-9
            t[t > 1] = 1
            valid[markers] = t
    elif not scaler and scale_type:
        scaler = get_scaler(scale_type)
        scaler.fit(train[markers])
        train[markers] = scaler.transform(train[markers])
        if valid is not None:
            valid[markers] = scaler.transform(valid[markers])
            t = valid[markers]
            t[t < 0] = 1e-9
            t[t > 1] = 1
            valid[markers] = t

    if add_root_values:
        train.loc[:, root_nodes] = input_value
        if valid is not None:
            valid.loc[:, root_nodes] = input_value

    if do_replace:
        for node in model.root_nodes:
            train.loc[train[node] == 0, node] = replace_value

    train_dict = train.to_dict("list")
    train_data = {
        k: DoubleTensor(v) for k, v in train_dict.items() if k in model.nodes()
    }
    train_inhibitors = {
        m1: DoubleTensor(
            [inhibition_value if m == m1 else 1.0 for m in train_dict["inhibitor"]]
        )
        for m1 in model.nodes()
    }
    train_input = {node: train_data[node] for node in model.root_nodes}

    if valid is not None:
        if do_replace:
            for node in model.root_nodes:
                valid.loc[valid[node] == 0, node] = replace_value
        valid_dict = valid.to_dict("list")

        valid_data = {
            k: DoubleTensor(v) for k, v in valid_dict.items() if k in model.nodes()
        }

        valid_inhibitors = {
            m1: DoubleTensor(
                [inhibition_value if m == m1 else 1.0 for m in valid_dict["inhibitor"]]
            )
            for m1 in model.nodes()
        }
        valid_input = {node: valid_data[node] for node in model.root_nodes}

        return (
            train_data,
            valid_data,
            train_inhibitors,
            valid_inhibitors,
            train_input,
            valid_input,
            train,
            valid,
            scaler,
        )
    else:
        return (train_data, train_inhibitors, train_input, train, scaler)


def inhibitor_mapping(reverse: bool = False):
    inhibitor_mapping = {
        "EGF": "None",
        "full": "None",
        "iEGFR": "EGFR",
        "iMEK": "MEK12",
        "iPI3K": "PI3K",
        "iPKC": "PKC",
        "imTOR": "mTOR",
    }

    if reverse:
        return {v: k for k, v in inhibitor_mapping.items()}
    else:
        return inhibitor_mapping


def data_to_nodes_mapping():
    data_to_nodes = {
        "EGF": "EGF",
        "SERUM": "SERUM",
        "b.CATENIN": "b-catenin",
        "cleavedCas": "cleavedCas",
        "p.4EBP1": "4EBP1",
        "p.Akt.Ser473.": "AKT_S473",
        "p.AKT.Thr308.": "AKT_T308",
        "p.AMPK": "AMPK",
        "p.BTK": "BTK",
        "p.CREB": "CREB",
        "p.ERK": "ERK12",
        "p.FAK": "FAK",
        "p.GSK3b": "GSK3B",
        "p.H3": "H3",
        "p.JNK": "JNK",
        "p.MAP2K3": "MAP3Ks",
        "p.MAPKAPK2": "MAPKAPK2",
        "p.MEK": "MEK12",
        "p.MKK3.MKK6": "MKK36",
        "p.MKK4": "MKK4",
        "p.NFkB": "NFkB",
        "p.p38": "p38",
        "p.p53": "p53",
        "p.p90RSK": "p90RSK",
        "p.PDPK1": "PDPK1",
        "p.PLCg2": "PLCg2",
        "p.RB": "RB",
        "p.S6": "S6",
        "p.S6K": "p70S6K",
        "p.SMAD23": "SMAD23",
        "p.SRC": "SRC",
        "p.STAT1": "STAT1",
        "p.STAT3": "STAT3",
        "p.STAT5": "STAT5",
    }

    return data_to_nodes
