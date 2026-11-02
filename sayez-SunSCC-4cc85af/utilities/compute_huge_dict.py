import os
import glob
import json

from copy import deepcopy

import clustering_utilities as c_utils

import concurrent.futures
from itertools import repeat

# from tqdm.notebook import tqdm
from tqdm import tqdm

import argparse
import time

def get_time_string():
    # get current time
    start_time = time.time()
    # format time to string for logging
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    time_str = f'[{time_str}]'
    return time_str

def main(args):
    # get current time


    # wl_dir = "/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019/all"  
    wl_dir = "/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/all"  
    wl_list = sorted(glob.glob(os.path.join(wl_dir, '**/*.FTS'),recursive=True))
    wl_basenames = [ os.path.basename(wl) for wl in wl_list ]

    # masks_dir = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019/feb2023/T425-T375-T325_fgbg'
    masks_dir = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/T425-T375-T325_fgbg'

    sqlite_db_path = "/globalscratch/users/n/s/nsayez/Classification_dataset/drawings_sqlite.sqlite"
    database = sqlite_db_path
    print( get_time_string(), f'num images in directory: {len(wl_list)}', )

    rotten_list = [
        
        37,38,39,40,52, 64,65,69,70,
        
        72,97,99,100,101,102,103,104,142,159,160,161,169,187,190,211,212,218,264,300,312,314,316,319,322,327,339,
        343,353,356,387,408,413,414,418,424,425,448,473,474,493,512,508,611,614,666,675,696,726,330,747,750,758,
        761,784,804,823,832,840,855,914,935,940,948,990,1013,
        
        1025,1039,1040,1089,1172,1303,1332,1345,1397,1409,1413,1414,1421,1440,1444,1468,1469,1488,1576,1646,1692,
        1735,1815,1840,1867,1893,1900,1905,1919,1924,1925,1930,1953,1969,1992,
        
        2007,2039,2043,2045,2049,2050,2078,2121,2133,2143,2185,2208,2220,2254,2266,2272,2298,2344,3262,3274,2375,
        2445,2454,2468,2492,2494,2495,2500,2501,2503,2516,2518,2536,2568,2574,2598,2604,2633,2635,2749,2763,2815,
        2818,2820,2821,2834,2835,2851,2857,2867,2896,2899,2848,2951,2952,2956,2964,2980,2981,2994,
        
        3018,3092,3093,3097,3099,3101,3106,3118,3122,3123,3124,3140,3148
        
        
    ]
    # rotten_list = [ ]
    # rotten_list = [                    
    #                     69,75,93,94,201,203,311,315,337,341,
    #                     403,409,420,441,613,615,668,726,743,
    #                     755,778,779,976,996,
                            
    #                     1036,1066,1081,1138,1255,1296,1337,1379,1398,1471,
    #                     1688,1735,1823,1925,1958,1989,1990,1995,
        
    #                     2018,2030,2073,2078,2104,2139,2151,2204,2205,2206,2230,2300,
    #                     2312,2325,2327,2349,2376,2452,2460,2461,2559,2673,2778,
    #                     2790,2793,2826,2836,2894,2897,2962,
        
    #                     3035,3142,3172,3178,3196,3217,3224,3242,3292,3296,
    #                     3312,3317,3335,3353,3389,3390,3415,3431,3510,3535,
    #                     3551,3571,3614,3664,3720,
    #             ]


    # root_dir = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019/feb2023'
    root_dir = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2'
    tmp = root_dir+'/wl_list2dbGroups_Classification.json'

    print(f'{get_time_string()}  opening huge db dict')
    huge_db_dict = { }
    with open(tmp, 'r') as f:
        huge_db_dict = json.load(f)
    print(f'{get_time_string()}  DONE opening huge db dict')

    # num_cpu = 20
    num_cpu = args.num_cpu
    # num_cpu = 40
    input_type = args.input_type
    assert input_type in ['confidence_map', 'mask']



    look_distance = args.look_distance  # How far to look for neighbours.
    kernel_bandwidthLon = args.kernel_bandwidthLon  # Longitude Kernel parameter.
    kernel_bandwidthLat = args.kernel_bandwidthLat  # Latitude Kernel parameter.
    n_iterations = args.n_iterations # Number of iterations



    cur_huge_dict = { } 

    print(f'{get_time_string()}  starting multiprocessing')

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_cpu)) as executor:
        for result_key, result_dict in tqdm(executor.map(c_utils.process_one_image, 
                                                            wl_list[:],
                                                            repeat(huge_db_dict),
                                                            repeat(deepcopy(cur_huge_dict)),
                                                            repeat(wl_list),
                                                            repeat(rotten_list),
                                                            repeat(masks_dir),
                                                            repeat(look_distance),
                                                            repeat(kernel_bandwidthLon),
                                                            repeat(kernel_bandwidthLat),
                                                            repeat(n_iterations),
                                                            repeat(input_type)
                                                            )):
            # print(result_key)
            if not len(list(result_dict.keys())) == 0:
                cur_huge_dict[result_key] = deepcopy(result_dict)
    print(f'{get_time_string()}  DONE multiprocessing')   

    # output_dir = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019/feb2023/param_optimization'
    output_dir = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/param_optimization'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = f'cur_dict_2002-19_dist{look_distance}_Lon{kernel_bandwidthLon}_lat{kernel_bandwidthLat}_iter{n_iterations}.json'
    print(f'{get_time_string()}  saving cur_huge_dict on disk')
    with open(os.path.join(output_dir, output_filename), 'w') as fp:
        json.dump(cur_huge_dict, fp, cls=c_utils.NpEncoder)
    print(f'{get_time_string()}  DONE saving cur_huge_dict on disk')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--look_distance', type=float, default=.1)
    parser.add_argument('--kernel_bandwidthLon', type=float, default=.25)
    parser.add_argument('--kernel_bandwidthLat', type=float, default=.08)
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--num_cpu', type=int, default=40)
    parser.add_argument('--input_type', type=str, default='mask')
    args = parser.parse_args()
    main(args)

