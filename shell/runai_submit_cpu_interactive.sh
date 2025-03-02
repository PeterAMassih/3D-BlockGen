arg_job_name="pabdel-cpu-int"

CLUSTER_USER="pabdel" # Your epfl username.
CLUSTER_USER_ID="226664"  # Your epfl UID.
CLUSTER_GROUP_NAME="ivrl" # Your group name.
CLUSTER_GROUP_ID="11227"  # Your epfl GID

# Change accordingly
#MY_IMAGE="ic-registry.epfl.ch/ivrl/ubuntu20-base" 
#MY_IMAGE="ic-registry.epfl.ch/ivrl/pytorch1.10:cuda11.3"
#MY_IMAGE="ic-registry.epfl.ch/ivrl/datascience-python"
MY_IMAGE="peteram/blockgen:cuda11.7v3"

echo "Job [$arg_job_name]"

runai submit $arg_job_name \
  -i $MY_IMAGE \
  -p ivrl-pabdel \
  --interactive \
  --cpu 10 --cpu-limit 14 \
  --memory 100G --memory-limit 100G \
  --gpu 0 \
  --node-pools g9 \
  --allow-privilege-escalation \
  --pvc runai-ivrl-pabdel-scratch:/scratch \
  --large-shm \
  --host-ipc \
  -e CLUSTER_USER=$CLUSTER_USER \
  -e CLUSTER_USER_ID=$CLUSTER_USER_ID \
  -e CLUSTER_GROUP_NAME=$CLUSTER_GROUP_NAME \
  -e CLUSTER_GROUP_ID=$CLUSTER_GROUP_ID \
  --command -- /bin/bash -c "sh /opt/lab/setup.sh && sleep infinity"

sleep 5 

runai describe job $arg_job_name -p ivrl-pabdel
