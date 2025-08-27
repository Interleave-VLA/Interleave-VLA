from .func import decode, decode_language_table

config = dict(
    bridge_dataset=dict(
        path_to_dataset="[PATH_TO_DATASET]/bridge_dataset/1.0.0",
        path_to_save="[PATH_TO_SAVE]/bridge_data_v2",
        get_prompt=lambda x: decode(x["language_instruction"]),
        get_observation=lambda x: x["observation"]["image_0"]
    ),
    fractal20220817_data=dict(
        path_to_dataset="[PATH_TO_DATASET]/fractal20220817_data/0.1.0", 
        path_to_save="[PATH_TO_SAVE]/fractal20220817_data",
        get_prompt=lambda x: decode(x['observation']['natural_language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    jaco_play=dict(
        path_to_dataset="[PATH_TO_DATASET]/jaco_play/0.1.0",
        path_to_save="[PATH_TO_SAVE]/jaco_play",
        get_prompt=lambda x: decode(x['observation']['natural_language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    stanford_hydra_dataset_converted_externally_to_rlds=dict(
        path_to_dataset="[PATH_TO_DATASET]/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0",
        path_to_save="[PATH_TO_SAVE]/stanford_hydra_dataset_converted_externally_to_rlds",
        get_prompt=lambda x: decode(x['language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    droid=dict(
        path_to_dataset="[PATH_TO_DATASET]/droid/1.0.0",
        path_to_save="[PATH_TO_SAVE]/droid",
        get_prompt=lambda x: decode(x['language_instruction']),
        get_observation=lambda x: x["observation"]["exterior_image_1_left"]
    ),
    bc_z=dict(  # v0.1.0
        path_to_dataset="[PATH_TO_DATASET]/bc_z/0.1.0",
        path_to_save="[PATH_TO_SAVE]/bc_z",
        get_prompt=lambda x: decode(x['observation']['natural_language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    berkeley_autolab_ur5=dict(
        path_to_dataset="[PATH_TO_DATASET]/berkeley_autolab_ur5/0.1.0",
        path_to_save="[PATH_TO_SAVE]/berkeley_autolab_ur5",
        get_prompt=lambda x: decode(x['observation']['natural_language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    iamlab_cmu_pickup_insert_converted_externally_to_rlds=dict(
        path_to_dataset="[PATH_TO_DATASET]/iamlab_cmu_pickup_insert_converted_externally_to_rlds/0.1.0",
        path_to_save="[PATH_TO_SAVE]/iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        get_prompt=lambda x: decode(x['language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    austin_sirius_dataset_converted_externally_to_rlds=dict(
        path_to_dataset="[PATH_TO_DATASET]/austin_sirius_dataset_converted_externally_to_rlds/0.1.0",
        path_to_save="[PATH_TO_SAVE]/austin_sirius_dataset_converted_externally_to_rlds",
        get_prompt=lambda x: decode(x['language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    ucsd_kitchen_dataset_converted_externally_to_rlds=dict(
        path_to_dataset="[PATH_TO_DATASET]/ucsd_kitchen_dataset_converted_externally_to_rlds/0.1.0",
        path_to_save="[PATH_TO_SAVE]/ucsd_kitchen_dataset_converted_externally_to_rlds",
        get_prompt=lambda x: decode(x['language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    ),
    language_table = dict(
        path_to_dataset="[PATH_TO_DATASET]/language_table/0.1.0",
        path_to_save="[PATH_TO_SAVE]/language_table",
        get_prompt=lambda x: decode_language_table(x['observation']['instruction']),
        get_observation=lambda x: x["observation"]["rgb"]
    ),
    utaustin_mutex=dict(
        path_to_dataset="[PATH_TO_DATASET]/utaustin_mutex/0.1.0",
        path_to_save="[PATH_TO_SAVE]/utaustin_mutex",
        get_prompt=lambda x: decode(x['language_instruction']),
        get_observation=lambda x: x["observation"]["image"]
    )
)

post_process_config = dict(
    bridge_dataset=dict(
        path_to_dataset="[PATH_TO_SAVE]/bridge_data_v2",
        path_to_save="[PATH_TO_SAVE]/bridge_data_v2_filtered",
    ),
    fractal20220817_data=dict(
        path_to_dataset="[PATH_TO_SAVE]/fractal20220817_data",
        path_to_save="[PATH_TO_SAVE]/fractal20220817_data_filtered",
    ),
    bc_z=dict(  # v0.1.0
        path_to_dataset="[PATH_TO_SAVE]/bc_z",
        path_to_save="[PATH_TO_SAVE]/bc_z_filtered",
    ),
    iamlab_cmu_pickup_insert_converted_externally_to_rlds=dict(
        path_to_dataset="[PATH_TO_SAVE]/iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        path_to_save="[PATH_TO_SAVE]/iamlab_cmu_pickup_insert_converted_externally_to_rlds_filtered",
    ),
    stanford_hydra_dataset_converted_externally_to_rlds=dict(
        path_to_dataset="[PATH_TO_SAVE]/stanford_hydra_dataset_converted_externally_to_rlds",
        path_to_save="[PATH_TO_SAVE]/stanford_hydra_dataset_converted_externally_to_rlds_filtered",
    ),
    jaco_play=dict(
        path_to_dataset="[PATH_TO_SAVE]/jaco_play",
        path_to_save="[PATH_TO_SAVE]/jaco_play_filtered",
    ),
    language_table=dict(
        path_to_dataset="[PATH_TO_SAVE]/language_table",
        path_to_save="[PATH_TO_SAVE]/language_table_filtered",
    ),
)

post_process_censor_objects = dict(
    bridge_dataset=["eggplant", "yellow cube", "green cube", "carrot", "plate", "spoon", "towel"], # basket
    fractal20220817_data=["rxbar"],
    bc_z=["eraser", "pepper"],
    iamlab_cmu_pickup_insert_converted_externally_to_rlds=["pink flower"],
    stanford_hydra_dataset_converted_externally_to_rlds=["toast"],
    jaco_play=["bread", "steak meat", "milk dairy"],
    language_table=[], # censor none, check all
)

model_path = dict(
    path_to_qwenvl="Qwen/Qwen2.5-VL-7B-Instruct",
    path_to_sam2_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    path_to_sam2="/path/to/sam2/checkpoints/sam2.1_hiera_large.pt", # please follow https://github.com/facebookresearch/sam2
    path_to_owlv2="google/owlv2-large-patch14-ensemble",
    path_to_qwen="Qwen/Qwen2.5-7B-Instruct",
)