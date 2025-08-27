def normalize(data_dict):
    total = sum(data_dict.values())
    return {k: v / total for k, v in data_dict.items()}

oxe_magic_soup_plus = {
    "fractal20220817_data": 0.194,  # Google RT-1 Robot Data (Large-Scale)
    # "kuka": 0.127, # distribute to fractal & bridge
    "bridge_dataset": 0.193,  # Original Version of Bridge V2 from Project Website
    # "taco_play": ,
    "jaco_play": 0.004,
    # "berkeley_cable_routing": 1.0,
    # "roboturk": 2.0,
    # "viola": 2.0,
    "berkeley_autolab_ur5": 0.012,
    # "toto": 1.0,
    "language_table": 0.044,
    "stanford_hydra_dataset_converted_externally_to_rlds": 0.044,
    # "austin_buds_dataset_converted_externally_to_rlds": 1.0,
    # "nyu_franka_play_dataset_converted_externally_to_rlds": 3.0,
    # "furniture_bench_dataset_converted_externally_to_rlds": 0.1,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 0.001,
    # "austin_sailor_dataset_converted_externally_to_rlds": 1.0,
    "austin_sirius_dataset_converted_externally_to_rlds": 0.017,
    # "dlr_edan_shared_control_converted_externally_to_rlds": 1.0,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 0.009,
    "utaustin_mutex": 0.022,
    # "berkeley_fanuc_manipulation": 2.0,
    # "cmu_stretch": 1.0,

    # New Datasets in MagicSoup++
    "bc_z": 0.075,  # Note: use v0.1.0 --> later versions broken
    # "fmb_dataset": 1.0,
    # "dobbe": 0.2,
    "droid": 0.05,  # Note: OpenVLA removed DROID from the data mixture for the final third of training
}

oxe_magic_soup_plus = normalize(oxe_magic_soup_plus)