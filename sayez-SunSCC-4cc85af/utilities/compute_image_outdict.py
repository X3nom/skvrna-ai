import os
import glob
import json

from copy import deepcopy

import clustering_utilities as c_utils
import matching_utilities as m_utils

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
    wl_dir = "/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/all"  
    wl_list = sorted(glob.glob(os.path.join(wl_dir, '**/*.FTS'),recursive=True))
    wl_basenames = [ os.path.basename(wl) for wl in wl_list ]

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
    show = False

    #####
    param_optim_folder = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/param_optimization'
    #####
    image_out_dict = {}
    image_out_dict_stats = {}

#     cur_huge_dict = deepcopy(grid_huge_dict[str(param_idx)])
    fn = f'cur_dict_2002-19_dist{args.look_distance}_Lon{args.kernel_bandwidthLon}_lat{args.kernel_bandwidthLat}_iter{args.n_iterations}.json' 
    # fn = f'cur_dict_2002-19_dist{params["look_distance"]}_Lon{params["kernel_bandwidthLon"]}_lat{params["kernel_bandwidthLat"]}_iter{params["n_iterations"]}.json'  
    print("Currently processing:")
    print(fn)

    print("Loading huge dict")
    cur_huge_dict_filename = os.path.join(param_optim_folder,fn)
    with open(cur_huge_dict_filename,'r') as f:
        cur_huge_dict = json.load(f)
        
    print(len(cur_huge_dict))

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_cpu)) as executor:
        for result_key, result_dict, result_dict_stats in tqdm(executor.map(m_utils.find_matchings_one_image, 
                                                            repeat(cur_huge_dict),
#                                                             list(cur_huge_dict.keys())[75:83],
                                                            # list(cur_huge_dict.keys())[2000:2005],
                                                            repeat(huge_db_dict),
                                                            list(cur_huge_dict.keys())[:],
                                                            repeat(wl_dir),
                                                            repeat(masks_dir),
                                                            repeat(input_type),
                                                            repeat(show)
                                                            )):
            # print(result_key)
#             print(result_dict_stats["ms_too_far"])
            image_out_dict[result_key] = result_dict
            image_out_dict_stats[result_key] = result_dict_stats
            # image_out_dict_stats[result_key] = result_dict['stats']
#             break
   
        
        

    print('num_images: ', len(list(image_out_dict.keys())))
    num_groups = 0
    for k,v in image_out_dict.items():
        # print(k,v)
        if v:
            num_groups += len(v['groups'])
    print("num_groups: ",num_groups)
    # print(image_out_dict)


    param_optim_folder2 = '/globalscratch/users/n/s/nsayez/Classification_dataset/2002-2019_2/param_optimP2'

    # saving the results in a json file
    out_file1 = os.path.join(param_optim_folder2, f'image_outdict_dist{args.look_distance}_Lon{args.kernel_bandwidthLon}_lat{args.kernel_bandwidthLat}_iter{args.n_iterations}.json')
    out_file2 = os.path.join(param_optim_folder2, f'image_outdictStats_dist{args.look_distance}_Lon{args.kernel_bandwidthLon}_lat{args.kernel_bandwidthLat}_iter{args.n_iterations}.json')

    with open(out_file1, 'w') as f:
        json.dump(image_out_dict, f, cls=c_utils.NpEncoder)
    with open(out_file2, 'w') as f:
        json.dump(image_out_dict_stats, f, cls=c_utils.NpEncoder)
        


      


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

