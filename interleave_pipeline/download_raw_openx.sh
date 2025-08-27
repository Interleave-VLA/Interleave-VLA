# ======= Install gsutil =========
# curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
# echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-cli -y
# ================================

DATA_DIR="/path/to/your/data/dir"

gsutil -m cp -r "gs://gresearch/robotics/jaco_play" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/berkeley_autolab_ur5" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/language_table" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/austin_sirius_dataset_converted_externally_to_rlds" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/bc_z" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/kuka" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/language_table" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/stanford_hydra_dataset_converted_externally_to_rlds" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/droid" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/ucsd_kitchen_dataset_converted_externally_to_rlds" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/iamlab_cmu_pickup_insert_converted_externally_to_rlds" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/austin_sirius_dataset_converted_externally_to_rlds" $DATA_DIR
gsutil -m cp -r "gs://gresearch/robotics/utaustin_mutex" $DATA_DIR
