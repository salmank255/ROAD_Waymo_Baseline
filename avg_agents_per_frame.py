
import json
import numpy as np
from statistics import mean 

road_trainval_path = '../road_waymo/road_waymo_trainval_v1.0.json'
road_test_path = '../road_waymo/road_waymo_test_v1.0.json'

row = 0
col = 0

# Agent labels count

all_agents_list = []
for city in ['location_phx','location_other','location_sf']:
    agents_list = []
    for d_path in [road_trainval_path,road_test_path]:
        with open(d_path,'r') as fff:
            road_json = json.load(fff)

        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                    agents_list.append(len(road_json['db'][videoname]['frames'][frame]['annos']))
                    all_agents_list.append(len(road_json['db'][videoname]['frames'][frame]['annos']))


    print(city, " averge agents per frame: ", mean(agents_list))

print("Road-Waymo averge agents per frame: ", mean(all_agents_list))