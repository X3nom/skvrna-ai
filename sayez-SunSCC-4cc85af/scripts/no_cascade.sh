#!/bin/bash

############## Parameters
#######S#B#A#T#C#H --cpus-per-task=40


#S#B#A#T#C#H --gres=gpu:TeslaA100_80:1

#SBATCH --gres=gpu:1
#SBATCH --constraint=TeslaA100_80|TeslaA100


# ATTENTION: the SBATCH --jobname is set in the calling script
# in the job_name variable (argindex 5)


#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=7G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --open-mode=append
#SBATCH --output="%x-%j.out"

#SBATCH --partition=gpu
#SBATCH --time=1800

#S#B#A#T#C#H --exclude=mb-cas001,mb-cas101,mb-mil102
#SBATCH --exclude=mb-cas001,mb-cas101,mb-mil102


###################################
# getthe arguments
random_move=$1
focus_on_group=$2
use_dtypes=$3
use_npy=$4
use_classes=$5

max_epochs=$6
parts_to_train=$7

lr=$8
T_max=$9
eta_min=${10}

batch_size=${11}
num_workers=${12}
char_to_balance=${13}

job_name=${14}

wandb_project=${15}

visual_input=${16}
l=$(echo $visual_input | tr -cd , | wc -c)
l2=$((l+1))
echo "visual_input: $visual_input, length = ${#visual_input}, length2 = ${l2}"

numeric_input=${17}
l=$(echo $numeric_input | tr -cd , | wc -c)
l3=$((l+1))


cascade=${18}
resnet_size=${19}

seed=${20}


echo "resnet_size: $resnet_size"

dt=$(date '+%Y-%m-%d/%H-%M-%S');
hydra_out_dir="./outputs/${dt}_${job_name}"

echo "hydra_out_dir: $hydra_out_dir"

echo "random_move: $random_move, focus_on_group: $focus_on_group, use_dtypes: $use_dtypes, use_npy: $use_npy, \n use_classes: $use_classes, max_epochs: $max_epochs, parts_to_train: $parts_to_train, \n lr: $lr, T_max: $T_max, eta_min: $eta_min, batch_size: $batch_size, \n num_workers: $num_workers, char_to_balance: $char_to_balance, job_name: $job_name, \n wandb_project: $wandb_project, visual_input: $visual_input"
echo "cascade: $cascade"
echo "seed: $seed"


# srun bioblue exp=Classification_Superclasses_fast gpus=1  \
bioblue exp=Classification_Superclasses_fast gpus=1  \
            hydra.run.dir=$hydra_out_dir \
            seed=$seed \
            gpus=1 \
            ++focus_on_group=$focus_on_group ++random_move=$random_move  \
            ++use_dtypes=$use_dtypes \
            ++use_npy=$use_npy \
            ++use_classes=$use_classes \
            ++first_classes='${use_classes}' \
            ++second_classes="['x','r','sym','asym']" \
            ++third_classes='["x","o","frag"]' \
            model=McIntoshGeneric ++trainer.max_epochs=$max_epochs \
            ++model.input_format.visual=$visual_input \
            ++model.input_format.numeric=$numeric_input \
            ++model.architecture.encoder.in_channels=$l2 \
            ++model.architecture.encoder.resnet_version=$resnet_size \
            ++model.cascade=$cascade \
            ++model.parts_to_train=$parts_to_train  \
            ++module.lr=$lr ++module.scheduler.T_max=$T_max  ++module.scheduler.eta_min=$eta_min \
            ++module.class1_weights="[1,1,1,1,1]" \
            ++module.class2_weights="[1,1,1,1]" \
            ++module.class3_weights="[1,1,1]" \
            ++dataset.num_workers=$num_workers ++dataset.batch_size=$batch_size \
            ++dataset.char_to_balance=$char_to_balance \
            ++logger.0.project=$wandb_project ++logger.0.name=$job_name


# move output file to wandb folder

output_file="$SLURM_JOB_NAME-$SLURM_JOB_ID.out"
echo $output_file

run_dir=$(find outputs/ -name  "$(grep 'wandb: Run data is saved locally in ./wandb/' $output_file | sed -r 's/^([^ ]+ ){7}.\/wandb\///')" | sed 's/wandb\/.*//' )
echo $run_dir

mv $output_file $run_dir

