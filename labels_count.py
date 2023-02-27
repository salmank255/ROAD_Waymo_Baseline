
import json
import numpy as np


road_trainval_path = '../roadpp/road_plus_plus_trainval_v1.0.json'
road_test_path = '../roadpp/road_plus_plus_test_v1.0.json'

# for d_path in [road_trainval_path,road_test_path]:
#     with open(d_path,'r') as fff:
#         road_json = json.load(fff)


#     agents = road_json['all_agent_labels']
#     agent_tubes = np.zeros(len(agents),dtype=int)
#     tube_set = set()
#     for videoname in road_json['db']:
#         for frame in road_json['db'][videoname]['frames']:
#             if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
#                 for anno in road_json['db'][videoname]['frames'][frame]['annos']:
#                     if road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid'] in tube_set:
#                         continue
#                     else:
#                         tube_set.add(road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid'])
#                         for agent in road_json['db'][videoname]['frames'][frame]['annos'][anno]['agent_ids']:
#                             agent_tubes[agent] += 1

#     print("All tubes:", d_path)
#     print("Total tubes:", sum(agent_tubes))
#     for i in range(len(agent_tubes)):
#         print(agents[i] +" : "+ str(agent_tubes[i]))


# Agent labels count

for d_path in [road_trainval_path,road_test_path]:
    with open(d_path,'r') as fff:
        road_json = json.load(fff)

    agents = road_json['all_agent_labels']
    agent_labs = np.zeros(len(agents),dtype=int)
    tube_set = set()
    for videoname in road_json['db']:
        for frame in road_json['db'][videoname]['frames']:
            if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                    agns = road_json['db'][videoname]['frames'][frame]['annos'][anno]['agent_ids']
                    for agn in agns:
                        agent_labs[agn] += 1

    print("All agent labs:", d_path)
    print("Total labels:", sum(agent_labs))
    for i in range(len(agent_labs)):
        print(agents[i] +" : "+ str(agent_labs[i]))


# action labels count

for d_path in [road_trainval_path,road_test_path]:
    with open(d_path,'r') as fff:
        road_json = json.load(fff)


    actions = road_json['all_action_labels']
    action_labs = np.zeros(len(actions),dtype=int)
    tube_set = set()
    for videoname in road_json['db']:
        for frame in road_json['db'][videoname]['frames']:
            if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                    actns = road_json['db'][videoname]['frames'][frame]['annos'][anno]['action_ids']
                    for actn in actns:
                        action_labs[actn] += 1

    print("All actions labs:", d_path)
    print("Total labels:", sum(action_labs))
    for i in range(len(action_labs)):
        print(actions[i] +" : "+ str(action_labs[i]))


# action labels count

for d_path in [road_trainval_path,road_test_path]:
    with open(d_path,'r') as fff:
        road_json = json.load(fff)

    locs = road_json['old_loc_labels']
    loc_labs = np.zeros(len(locs),dtype=int)
    tube_set = set()
    for videoname in road_json['db']:
        for frame in road_json['db'][videoname]['frames']:
            if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                    loccs = road_json['db'][videoname]['frames'][frame]['annos'][anno]['loc_ids']
                    for loc in loccs:
                        loc_labs[loc] += 1

    print("All location labs:", d_path)
    print("Total labels:", sum(loc_labs))
    for i in range(len(loc_labs)):
        print(locs[i] +" : "+ str(loc_labs[i]))

