#!/bin/bash

if ! command -v nvidia-smi &> /dev/null
then
    echo "nvidia-smi not found, please install NVIDIA drivers."
    exit 1
fi

# use nvidia-smi to get the number of GPUs
np=$(nvidia-smi -L | wc -l)
ip_addr="127.0.0.1"
echo "GPU number: $np"

temp_dir=$(mktemp -d)
hostfile="$temp_dir/hostfile"

echo "127.0.0.1 slots=$np" > "$hostfile"
echo "Temp hostfile: $hostfile"

Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)

method=$1 # e.g., 3dmaster_part or 3dmaster_part_s2
name=$2 # e.g., debug_aaa or debug_aaa_s2
METHOD_NAME="$method"
CONFIG_EXP="$name"
mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-p ${Port}"  \
        -hostfile $hostfile \
        -x HOROVOD_MPI_THREADS_DISABLE=1 \
        -x MPI_THREAD_SINGLE=1 \
		-bind-to none  -map-by slot \
        --mca btl tcp,self \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_IB_GID_INDEX=3 \
        -x NCCL_MIN_NCHANNELS=16 \
        -x NCCL_IB_HCA=mlx5 \
        -x NCCL_IB_QPS_PER_CONNECTION=4 \
        -x NCCL_DEBUG=WARN \
		python main.py "$METHOD_NAME" --experiment-name "$CONFIG_EXP" "${@:3}" \
        2>&1 | tee logs/"$method"_"$name"_$(date +%Y.%m.%d_%H:%M:%S).log
