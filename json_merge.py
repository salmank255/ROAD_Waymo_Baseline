from mergedeep import merge
import json

anno_file = ['../road/road_test_v1.0.json', '../road_waymo/road_waymo_test_v1.0.json']

with open(anno_file[0], 'r') as fff:
    final_annots1 = json.load(fff)
with open(anno_file[1], 'r') as fff:
    final_annots2 = json.load(fff)

final_annots = merge(final_annots1, final_annots2) 


for vname in final_annots['db']:
    print(vname)
