#! /bin/bash
export MLFLOW_TRACKING_URI=$3

# Get current timestamp for experiment naming
printf -v now '%(%F_%H-%M-%S)T' -1
exp_name="tsc_pruned_$1_$now"
exp_create_str=$(mlflow experiments create -n $exp_name)
echo $exp_create_str
exp_id=$(echo $exp_create_str | awk '{print $NF}')

# Define model architectures and datasets to iterate over
models=("Hydrant" "Quant" "Hydra")
prune=("0.0" "0.8")
prune2=("0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.9" "0.95" "0.99")
datasets=("Pedestrian" "WISDM" "UCIActivity" "LakeIce" "Tiselac" "InsectSound" "USCActivity" "FordChallenge" "CrowdSourced" "WISDM2" "Skoda" "STEW" "AudioMNIST-DS" "CornellWhaleChallenge" "FruitFlies" "Opportunity" "PAMAP2" "WhaleSounds" "DREAMERA" "DREAMERV")

for d in "${datasets[@]}"
do
    for f in "0" "1" "2" "3" "4"
    do
        for m in "${models[@]}"
        do
            # only store 0% and 80% models
            for p in "${prune[@]}"
            do
                echo "Running model $m on GPU $1 on DS $d with BS $b for fold $f ..."
                timeout 14400 mlflow run --experiment-name=$exp_name -e main.py -P gpu=$1 -P cache_dir=$2 -P dataset=$d -P model=$m -P fold=$f -P prune_rate=$p ./ts_archive_experiments
            done
            mlflow experiments csv -x $exp_id > "$exp_name.csv"
            # test other pruning rates for ablation study
            for p in "${prune2[@]}"
            do
                echo "Running model $m on GPU $1 on DS $d with BS $b for fold $f ..."
                timeout 14400 mlflow run --experiment-name=$exp_name -e main.py -P gpu=$1 -P cache_dir=$2 -P dataset=$d -P model=$m -P fold=$f -P prune_rate=$p -P discard_model=True ./ts_archive_experiments
            done
            mlflow experiments csv -x $exp_id > "$exp_name.csv"
        done
    done
done