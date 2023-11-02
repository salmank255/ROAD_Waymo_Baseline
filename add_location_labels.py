
import json
import numpy as np
import csv

def find_in_list_of_list(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))


road_trainval_path = '../road_waymo/road_waymo_trainval_v1.0.json'
with open(road_trainval_path,'r') as fff:
    road_json = json.load(fff)

file = open('location_trainval.csv')
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    rows.append(row)


for vid in road_json['db']:
    vid_ind = find_in_list_of_list(rows, vid)[0]
    city = rows[vid_ind][1]
    road_json['db'][vid]['location'] = city

with open('../road_waymo/road_waymo_trainval_v1.1.json', 'w') as f:
    json.dump(road_json, f)




road_test_path = '../road_waymo/road_waymo_test_v1.0.json'
with open(road_test_path,'r') as fff:
    road_json = json.load(fff)

file = open('location_test.csv')
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    rows.append(row)

for vid in road_json['db']:
    vid_ind = find_in_list_of_list(rows, vid)[0]
    city = rows[vid_ind][1]
    road_json['db'][vid]['location'] = city

with open('../road_waymo/road_waymo_test_v1.1.json', 'w') as f:
    json.dump(road_json, f)


