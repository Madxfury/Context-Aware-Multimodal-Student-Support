#!/bin/bash
# Runs every job the report needs on this Mac (PyTorch MPS). Usage, from Terminal:
#   cd ~/"LY Project Samvya" && bash report_jobs/run_on_mac.sh
# Everything is written to report_jobs/outputs/. Safe to re-run: finished steps are skipped.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
source .venv/bin/activate
OUT=report_jobs/outputs; mkdir -p "$OUT"
{ date; sw_vers 2>/dev/null; sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null; python -c "import torch,timm;print('torch',torch.__version__,'timm',timm.__version__,'mps',torch.backends.mps.is_available())"; } > "$OUT/env_info.txt" 2>&1
step() { echo; echo "=== $1 ($(date +%H:%M))"; }

if [ ! -d "$HOME/Downloads/Samvya_Dataset/1_facial_emotion_FANE/images" ]; then
  step "Unzipping FANE images"; unzip -q -o "$HOME/Downloads/Samvya_Dataset_Full.zip" -d "$HOME/Downloads"
fi

step "1/4 Original Swin-FANE, HSEmotion and Grad-CAM on the 406 unseen test images"
[ -f "$OUT/fane_eval_original_status.json" ] || caffeinate -i python report_jobs/fane_eval_original.py 2>&1 | tee "$OUT/fane_eval_original.log"

step "2/4 WESAD download (2.25 GB) and preprocessing"
[ -f DataSet/WESAD/WESAD/S2/S2.pkl ] || caffeinate -i python "$HOME/Downloads/Samvya_Dataset/4_wesad_stress/download_wesad.py" 2>&1 | tee "$OUT/wesad_download.log"
[ -f DataSet/WESAD_Processed_V2/test.pt ] || caffeinate -i python scripts/preprocess_wesad_v2.py 2>&1 | tee "$OUT/wesad_preprocess.log"

step "3/4 WESAD LSTM vs BiLSTM: project script unchanged, then the logged re-run"
[ -f "$OUT/wesad_train_original_script.log" ] || caffeinate -i python scripts/train_wesad_models.py 2>&1 | tee "$OUT/wesad_train_original_script.log"
[ -f Models/Physiological_V2/training_metadata.json ] && cp Models/Physiological_V2/training_metadata.json "$OUT/wesad_training_metadata_original_script.json"
[ -f "$OUT/wesad_report_info.json" ] || caffeinate -i python report_jobs/wesad_logged.py 2>&1 | tee "$OUT/wesad_logged.log"

step "4/4 Retraining Swin-FANE on the deduplicated split (longest step)"
[ -f "$OUT/fane_retrain_info.json" ] || caffeinate -i python report_jobs/fane_retrain.py 2>&1 | tee "$OUT/fane_retrain.log"

step "Finished. Outputs are in report_jobs/outputs"
ls -la "$OUT"
