# !/bin/bash
model_names=("mocov3_vitb")

# iterate over different model names
export PYTHONPATH='/iris/u/kayburns/new_arch/mvp:/iris/u/kayburns/new_arch/moco-v3'
for model_name in "${model_names[@]}"
do
    echo "Evaluating model: $model_name"
    python evaluate_segmentation.py \
        --model_name $model_name
done