set -euo pipefail

# Generic one-command launcher for the JAX behavior cloning policy.
# Configure paths with environment variables instead of editing this file.

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DIR="${TRAIN_DIR:-data/example_task/demos/success/train}"
TEST_DIR="${TEST_DIR:-data/example_task/demos/success/test}"
VAE_CKPT="${VAE_CKPT:-pretrained_models/jax_ckpt/hand_vae}"
RESNET_PATH="${RESNET_PATH:-pretrained_models/resnet-18}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/behavior_clone_example}"

"${PYTHON_BIN}" imitation_learning/behavior_clone/scripts/train_jax.py "$@" \
  --train_dir "${TRAIN_DIR}" \
  --test_dir "${TEST_DIR}" \
  --vae_ckpt "${VAE_CKPT}" \
  --resnet_path "${RESNET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --backbone_impl hf_resnet18 \
  --hand_prior_source vae \
  --state_encoder mlp \
  --batch_size "${BATCH_SIZE:-128}" \
  --total_steps "${TOTAL_STEPS:-20000}" \
  --warmup_steps "${WARMUP_STEPS:-1000}" \
  --print_freq "${PRINT_FREQ:-100}" \
  --eval_freq "${EVAL_FREQ:-1000}" \
  --save_freq "${SAVE_FREQ:-5000}" \
  --seed "${SEED:-42}"
