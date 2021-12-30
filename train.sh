# IP address of master node
MASTER_ADDR=10.46.178.157

# port of master node
MASTER_PORT=28625

# rank of node，the master node must be 0
node_rank=$1

# number of node
nnodes=$2

# number of GPU per node
nproc_per_node=2

# avoid NCCL errors, https://github.com/PyTorchLightning/pytorch-lightning/issues/4420
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

# start command
DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --node_rank $node_rank --nnodes $nnodes --master_addr $MASTER_ADDR --master_port $MASTER_PORT" CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS main.py
