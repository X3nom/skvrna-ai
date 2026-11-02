#!/bin/bash

# run_dirs=('/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/21-54-59_WL-Excentricity_noNumeric_class1_100epochs_run21' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/21-55-02_WL-Excentricity_noNumeric_class1_100epochs_run22' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/21-55-20_WL-Excentricity_noNumeric_class1_100epochs_run23' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/21-55-20_WL-Excentricity_noNumeric_class1_100epochs_run24' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/21-55-20_WL-Excentricity_noNumeric_class1_100epochs_run25' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/21-55-20_WL-Excentricity_noNumeric_class1_100epochs_run26' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/22-33-05_WL-Excentricity_noNumeric_class1_100epochs_run27' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/22-36-36_WL-Excentricity_noNumeric_class1_100epochs_run28' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/22-51-29_WL-Excentricity_noNumeric_class1_100epochs_run29' \
# '/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-06/22-52-11_WL-Excentricity_noNumeric_class1_100epochs_run30')

run_dirs=('/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-04-00_WL-Excentricity_noNumeric_class1_100epochs_run31' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-04-00_WL-Excentricity_noNumeric_class1_100epochs_run32' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-04-02_WL-Excentricity_noNumeric_class1_100epochs_run33' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-04-02_WL-Excentricity_noNumeric_class1_100epochs_run34' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-46-54_WL-Excentricity_noNumeric_class1_100epochs_run35' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-59-33_WL-Excentricity_noNumeric_class1_100epochs_run36' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/00-59-38_WL-Excentricity_noNumeric_class1_100epochs_run37' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/01-19-52_WL-Excentricity_noNumeric_class1_100epochs_run38' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/01-29-30_WL-Excentricity_noNumeric_class1_100epochs_run39' \
'/home/ucl/elen/nsayez/bio-blueprints/outputs/2023-03-07/01-55-54_WL-Excentricity_noNumeric_class1_100epochs_run40')

for run_dir in "${run_dirs[@]}"
do
    echo "run_dir: $run_dir"
    sbatch /home/ucl/elen/nsayez/bio-blueprints/scripts/TrainP2_P3.sh $run_dir
done