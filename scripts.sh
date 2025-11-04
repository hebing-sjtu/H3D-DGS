#!/bin/bash
# run_all_parallel.sh
# Automatically detect GPUs and run training, rendering, and evaluation tasks in parallel

# ===============================
# Configuration
# ===============================
# Neural_seq_list=('sear_steak' 'cook_spinach' 'coffee_martini' 'cut_roasted_beef' 'flame_salmon_1' 'flame_steak')
# CMU_seq_list=('basketball' 'softball' 'boxes')
Neural_seq_list=('sear_steak')
CMU_seq_list=()
DATASET_NEURAL="/root/H3D-DGS/datasets/Neural3D"
DATASET_CMU="/root/H3D-DGS/datasets/CMU-Panoptic"

# ===============================
# Detect available GPUs
# ===============================
NUM_GPUS=$(nvidia-smi -L | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ No GPU devices detected. Exiting."
    exit 1
fi
echo "✅ Detected $NUM_GPUS GPU(s)."

# ===============================
# Function to run one task
# ===============================
run_task() {
    cmd=$1
    cuda_id=$2
    seq_name=$3
    echo "▶️ Launching task: $seq_name on GPU $cuda_id"
    $cmd --cuda_id "$cuda_id" > logs/${seq_name}.log 2>&1 &
}

# Create log directory
mkdir -p logs

# ===============================
# 1️⃣ Training phase
# ===============================
all_seqs=()
for seq in "${Neural_seq_list[@]}"; do
    all_seqs+=("python demo/train.py --dataset_dir $DATASET_NEURAL --sequence $seq")
done
for seq in "${CMU_seq_list[@]}"; do
    all_seqs+=("python demo/train.py --dataset_dir $DATASET_CMU --sequence $seq --is_sport")
done

echo "=== Starting training phase ==="
i=0
for cmd in "${all_seqs[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    seq_name=$(echo "$cmd" | grep -oP '(?<=--sequence )\S+')
    run_task "$cmd" "$gpu_id" "$seq_name"
    ((i++))
done
wait
echo "✅ Training finished."
exit 0

# ===============================
# 2️⃣ Rendering phase
# ===============================
all_seqs=()
for seq in "${Neural_seq_list[@]}"; do
    all_seqs+=("python demo/render.py --dataset_dir $DATASET_NEURAL --sequence $seq")
done
for seq in "${CMU_seq_list[@]}"; do
    all_seqs+=("python demo/render.py --dataset_dir $DATASET_CMU --sequence $seq --is_sport")
done

echo "=== Starting rendering phase ==="
i=0
for cmd in "${all_seqs[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    seq_name=$(echo "$cmd" | grep -oP '(?<=--sequence )\S+')
    run_task "$cmd" "$gpu_id" "$seq_name"
    ((i++))
done
wait
echo "✅ Rendering finished."

# ===============================
# 3️⃣ Evaluation phase
# ===============================
all_seqs=()
for seq in "${Neural_seq_list[@]}"; do
    all_seqs+=("python demo/eval.py --dataset_dir $DATASET_NEURAL --sequence $seq")
done
for seq in "${CMU_seq_list[@]}"; do
    all_seqs+=("python demo/eval.py --dataset_dir $DATASET_CMU --sequence $seq --is_sport")
done

echo "=== Starting evaluation phase ==="
i=0
for cmd in "${all_seqs[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    seq_name=$(echo "$cmd" | grep -oP '(?<=--sequence )\S+')
    run_task "$cmd" "$gpu_id" "$seq_name"
    ((i++))
done
wait
echo "✅ All tasks completed."