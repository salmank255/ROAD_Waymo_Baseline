
import json
import numpy as np

import xlsxwriter
 
workbook = xlsxwriter.Workbook('label_count.xlsx')
 
# By default worksheet names in the spreadsheet will be
# Sheet1, Sheet2 etc., but we can also specify a name.
worksheet = workbook.add_worksheet("My sheet")


road_trainval_path = '../road_waymo/road_waymo_trainval_v1.0.json'
road_test_path = '../road_waymo/road_waymo_test_v1.0.json'

row = 0
col = 0

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



# AV action labels count

for d_path in [road_trainval_path,road_test_path]:
    with open(d_path,'r') as fff:
        road_json = json.load(fff)


    av_actions = road_json['all_av_action_labels']
    av_action_labs = np.zeros(len(av_actions),dtype=int)
    for videoname in road_json['db']:
        for frame in road_json['db'][videoname]['frames']:
            actns = road_json['db'][videoname]['frames'][frame]['av_action_ids']
            for actn in actns:
                av_action_labs[actn] += 1

    print("All AV actions labs:", d_path)
    worksheet.write(row, col, d_path)
    row += 1  
    print("Total labels:", sum(av_action_labs))
    for i in range(len(av_action_labs)):
        print(av_actions[i] +" : "+ str(av_action_labs[i]))
        worksheet.write(row, col, av_actions[i])
        worksheet.write(row, col + 1, str(av_action_labs[i]))
        row += 1


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
    worksheet.write(row, col, d_path)
    row += 1    
    print("Total labels:", sum(agent_labs))
    for i in range(len(agent_labs)):
        print(agents[i] +" : "+ str(agent_labs[i]))
        worksheet.write(row, col, agents[i])
        worksheet.write(row, col + 1, str(agent_labs[i]))
        row += 1


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
    worksheet.write(row, col, d_path)
    row += 1  
    print("Total labels:", sum(action_labs))
    for i in range(len(action_labs)):
        print(actions[i] +" : "+ str(action_labs[i]))
        worksheet.write(row, col, actions[i])
        worksheet.write(row, col + 1, str(action_labs[i]))
        row += 1

# location labels count

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
    worksheet.write(row, col, d_path)
    row += 1  
    print("Total labels:", sum(loc_labs))
    for i in range(len(loc_labs)):
        print(locs[i] +" : "+ str(loc_labs[i]))
        worksheet.write(row, col, locs[i])
        worksheet.write(row, col + 1, str(loc_labs[i]))
        row += 1


# duplex labels count

for d_path in [road_trainval_path,road_test_path]:
    dup_final_list = []
    with open(d_path,'r') as fff:
        road_json = json.load(fff)

    dups = road_json['all_duplex_labels']
    dup_labs = np.zeros(len(dups),dtype=int)
    tube_set = set()
    for videoname in road_json['db']:
        for frame in road_json['db'][videoname]['frames']:
            if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                    dupss = road_json['db'][videoname]['frames'][frame]['annos'][anno]['duplex_ids']
                    for dup in dupss:
                        dup_labs[dup] += 1

    print("All duplex labs:", d_path)
    worksheet.write(row, col, d_path)
    row += 1  
    print("Total labels:", sum(dup_labs))
    for i in range(len(dup_labs)):
        if dup_labs[i] >500:
            print(dups[i] +" : "+ str(dup_labs[i]))
            dup_final_list.append(dups[i])
            worksheet.write(row, col, dups[i])
            worksheet.write(row, col + 1, str(dup_labs[i]))
            row += 1

    print(dup_final_list)


# triplet labels count

for d_path in [road_trainval_path,road_test_path]:
    trip_final_list = []
    with open(d_path,'r') as fff:
        road_json = json.load(fff)

    trips = road_json['all_triplet_labels']
    trip_labs = np.zeros(len(trips),dtype=int)
    tube_set = set()
    for videoname in road_json['db']:
        for frame in road_json['db'][videoname]['frames']:
            if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                    tripss = road_json['db'][videoname]['frames'][frame]['annos'][anno]['triplet_ids']
                    for trip in tripss:
                        trip_labs[trip] += 1

    print("All triplet labs:", d_path)
    worksheet.write(row, col, d_path)
    row += 1  
    print("Total labels:", sum(trip_labs))
    for i in range(len(trip_labs)):
        if trip_labs[i] >500:
            print(trips[i] +" : "+ str(trip_labs[i]))
            trip_final_list.append(trips[i])
            worksheet.write(row, col, trips[i])
            worksheet.write(row, col + 1, str(trip_labs[i]))
            row += 1

    print(trip_final_list)

workbook.close()