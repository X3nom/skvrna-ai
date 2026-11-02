#!/bin/bash

function join_by { local IFS="$1"; shift; echo "$*"; }

num_workers=16

# all_single_thresh=(['image','2023_T500_fgbg'])
# all_single_thresh=(['image','2023_T450_fgbg'])
# all_single_thresh=(['image','2023_T400_fgbg'])
# all_single_thresh=(['image','2023_T350_fgbg'])
# all_single_thresh=(['image','2023_T300_fgbg'])
# all_single_thresh=(['image','2023_T500_fgbg'] ['image','2023_T450_fgbg'] ['image','2023_T400_fgbg'] ['image','2023_T350_fgbg'] ['image','2023_T300_fgbg'])
all_single_thresh=(['image','2023_T475_fgbg'] ['image','2023_T425_fgbg'] ['image','2023_T375_fgbg'] ['image','2023_T325_fgbg'] )
# all_single_thresh=(['image','2023_T475_fgbg'] )
# all_single_thresh=(['image','2023_T425_fgbg'] )
# all_single_thresh=(['image','2023_T375_fgbg'] )
# all_single_thresh=(['image','2023_T325_fgbg'] )
max_epochs=40
project_name="2023_SingleThreshold_fgbg_correct"
# project_name="2023_Deepsun"

# run_number=0
# run_number=2
# run_number=3
# run_number=4
run_number=5

scheduler_type='StepLR' # StepLR or MultistepLR or None
scheduler_interval='epoch' # used for MultistepLR scheduler and StepLR scheduler
# scheduler_interval='step' #  CANNOT BE USED FOR MultistepLR scheduler
scheduler_step_size=1 # used for StepLR scheduler, don't use for MultistepLR scheduler
classes=["sunspot"]

for use_dtype in ${all_single_thresh[@]}; do
    # echo ${i}
    # echo ${i} | sed -e 's/.*\[\(T[0-9]+/\1/' # use regular expressions to get the threshold value in i
    # NUMBER=$(echo "The store is 12 miles away plus 32." | grep -o -E '[0-9]+') ; echo $NUMBER
    NUMBER=$(echo "${use_dtype}" | grep -o -E 'T[0-9]+')
    id="UNet_$(join_by '_' ${NUMBER})"
    # echo "$id"
    # use regular expressions to get the threshold value in i
    
    cd ~
    # echo $PWD
    
    if [ "$scheduler_type" = "MultistepLR" ]; then
        # MultistepLR scheduler (milestones=[20, 30] epochs)    
        job_name="2013-15_${id}_MultistepLR_${scheduler_step_size}"
    elif [ "$scheduler_type" = "StepLR" ]; then
        # StepLR scheduler (step_size=1 epoch)
        job_name="2013-15_${id}_StepLR_${scheduler_interval}_${scheduler_step_size}"
    else
        # No LR scheduler = StepLR scheduler (step_size> max_epochs)
        job_name="2013-15_${id}_NoLRscheduler"
    fi

    job_name="${job_name}_run${run_number}"

    # job_name="2013-15_${id}_MultistepLR" # MultistepLR scheduler (milestones=[20, 30] epochs) (DONE for scheduler_interval='epoch')
    # job_name="2013-15_${id}" # No LR scheduler (DONE for scheduler_interval='epoch')
    # echo "$job_name"


    echo "sbatch Segmentation_slurm_launcher.sh workers=$num_workers classes=$classes use_dtype=$use_dtype num_epochs=$max_epochs project_name=$project_name job_name=$job_name, scheduler_type=$scheduler_type scheduler_interval=$scheduler_interval, scheduler_step_size=$scheduler_step_size"
    sbatch ./bio-blueprints/scripts/Segmentation_slurm_launcher.sh $num_workers $use_dtype $max_epochs $project_name $job_name $scheduler_type $scheduler_interval $scheduler_step_size $run_number $classes
done


