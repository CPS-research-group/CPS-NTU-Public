#!/bin/bash
# Train All Instances of the BetaVAE detector for paper.

EXP_NO="7A"
DATAPATH="/media/diskb/michaelj004"
TRAINPATH="${DATAPATH}/Train${EXP_NO}"
BETA=1.4
N_LATENT=30
BATCH=16
SIZES=("224x224" "112x112" "56x56" "28x28" "14x14" "7x7")

for size in ${SIZES[@]}; do
  echo "Starting trainig with image size $size..."
  python bvae.py train \
    --beta $BETA \
    --n_latent $N_LATENT \
    --dimensions $size \
    --dataset $TRAINPATH \
    --batch $BATCH
  python bvae.py train \
    --beta $BETA \
    --n_latent $N_LATENT \
    --dimensions $size \
    --grayscale \
    --dataset $TRAINPATH \
    --batch $BATCH
done

# Generate encoder only model.
for size in ${SIZES[@]}; do
  echo "Converting $size..."
  python bvae.py \
    --weights "bvae_n${N_LATENT}_b${BETA}__${size}.pt" convert
  python bvae.py \
    --weights "bvae_n${N_LATENT}_b${BETA}_bw_${size}.pt" convert
done

# Calibrate model and determine post-processing hyperparameters.
CALIBRATIONPATH="${DATAPATH}/Calibration${EXP_NO}"
for size in ${SIZES[@]}; do
  echo "Beginning calibration for ${size}"
  python calibration.py \
    --weights "bvae_n${N_LATENT}_b${BETA}__${size}.pt" \
    --dataset $CALIBRATIONPATH
  python calibration.py \
    --weights "bvae_n${N_LATENT}_b${BETA}_bw_${size}.pt" \
    --dataset $CALIBRATIONPATH
done

# Find optimal decay term for CUMSUM.
test_vids=$(ls "${DATAPATH}\Test${EXP_NO}\*.avi")
for video in ${test_vids[@]}; do
  for size in ${SIZES[@]}; do
    echo "Processing ${video} with ${size} network.."
    python test.py \
      --weights "enc_only_bvae_n${N_LATENT}_b${BETA}__${size}.pt" \
      --alpha_cal "alpha_cal_bvae_n${N_LATENT}_b${BETA}__${size}.json" \
      --video $video
    python test.py \
      --weights "enc_only_bvae_n${N_LATENT}_b${BETA}_bw_${size}.pt" \
      --alpha_cal "alpha_cal_bvae_n${N_LATENT}_b${BETA}__${size}.json" \
      --video $video
  done
done

for size in ${SIZES[@]}; do
  echo "Finding optimal decay for ${size}"
  results_id=$(ls "*_${size}_scene?_empty?_window*.xlsx")
  results_id_bw=$(ls "*_bw_${size}_scene?_empty?_window*.xlsx")
  results_list_rain=$(ls "*__${size}_*_brightness0.0_*.xlsx")
  results_list_rain_bw=$(ls "*_bw_${size}_*_brightness0.0_*.xlsx")
  results_list_brightness=$(ls "*__${size}_*_rain0.0_*.xlsx")
  results_list_brightness_bw=$(ls "*_bw_${size}_*_rain0.0_*.xlsx")
  python find_optimal_decay.py $results_id $results_list_rain \
    --cal "alpha_cal_bvae_n${N_LATENT}_b${BETA}__${size}.json" \
    --partition rain
  python find_optimal_decay.py $results_id_bw $results_list_rain_bw \
    --cal "alpha_cal_bvae_n${N_LATENT}_b${BETA}_bw_${size}.json" \
    --partition rain
  python find_optimal_decay.py $results_id $results_list_brightness \
    --cal "alpha_cal_bvae_n${N_LATENT}_b${BETA}__${size}.json" \
    --partition brightness
  python find_optimal_decay.py $results_id_bw $results_list_brightness_bw \
    --cal "alpha_cal_bvae_n${N_LATENT}_b${BETA}_bw_${size}.json" \
    --partition brightness
done

echo "Done :)"
