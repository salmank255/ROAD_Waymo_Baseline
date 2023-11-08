
import json
import numpy as np

import xlsxwriter
 
workbook = xlsxwriter.Workbook('label_count_cities.xlsx')
 
# By default worksheet names in the spreadsheet will be
# Sheet1, Sheet2 etc., but we can also specify a name.
worksheet = workbook.add_worksheet("My sheet")


road_trainval_path = '../road_waymo/road_waymo_trainval_v1.1.json'
road_test_path = '../road_waymo/road_waymo_test_v1.1.json'

row = 0
col = 0


# AV action labels count

for d_path in [road_trainval_path,road_test_path]:
    with open(d_path,'r') as fff:
        road_json = json.load(fff)


    av_actions = road_json['all_av_action_labels']
    for city in ['location_phx','location_other','location_sf']:
        av_action_labs = np.zeros(len(av_actions),dtype=int)
        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                actns = road_json['db'][videoname]['frames'][frame]['av_action_ids']
                for actn in actns:
                    av_action_labs[actn] += 1

        print(city+" AV actions labs: "+ d_path)
        worksheet.write(row, col, city+" AV actions labs: "+ d_path)
        row += 1
        worksheet.write(row, col, "Total: ")
        worksheet.write(row, col+1, sum(av_action_labs))
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

    
    for city in ['location_phx','location_other','location_sf']:
        tube_set = set()
        agent_labs = np.zeros(len(agents),dtype=int)
        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                    for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                        agns = road_json['db'][videoname]['frames'][frame]['annos'][anno]['agent_ids']
                        tube_id =  road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid']
                        for agn in agns:
                            agent_labs[agn] += 1
                        tube_set.add(tube_id)

        print(city+ " agent labs: "+ d_path)
        worksheet.write(row, col, city+ " agent labs: "+ d_path)
        row += 1
        worksheet.write(row, col, "Tubes: ")
        worksheet.write(row, col+1, len(tube_set))
        row += 1    
        worksheet.write(row, col, "Total: ")
        worksheet.write(row, col+1, sum(agent_labs))
        row += 1
        print("Total Tubes:", len(tube_set))
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
    
    
    for city in ['location_phx','location_other','location_sf']:
        tube_set = set()
        action_labs = np.zeros(len(actions),dtype=int)
        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                    for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                        actns = road_json['db'][videoname]['frames'][frame]['annos'][anno]['action_ids']
                        tube_id =  road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid']
                        for actn in actns:
                            action_labs[actn] += 1
                        tube_set.add(tube_id)
        print(city+"  actions labs: "+ d_path)
        worksheet.write(row, col, city+"  actions labs: "+ d_path)
        row += 1
        worksheet.write(row, col, "Tubes: ")
        worksheet.write(row, col+1, len(tube_set))
        row += 1   
        worksheet.write(row, col, "Total: ")
        worksheet.write(row, col+1, sum(action_labs))
        row += 1 
        print("Total Tubes:", len(tube_set)) 
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
    
    
    for city in ['location_phx','location_other','location_sf']:
        tube_set = set()
        loc_labs = np.zeros(len(locs),dtype=int)
        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                    for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                        loccs = road_json['db'][videoname]['frames'][frame]['annos'][anno]['loc_ids']
                        tube_id =  road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid']
                        for loc in loccs:
                            loc_labs[loc] += 1
                        tube_set.add(tube_id)
        print(city+"  location labs: "+ d_path)
        worksheet.write(row, col, city+"  location labs: "+ d_path)
        row += 1
        worksheet.write(row, col, "Tubes: ")
        worksheet.write(row, col+1, len(tube_set))
        row += 1   
        worksheet.write(row, col, "Total: ")
        worksheet.write(row, col+1, sum(loc_labs))
        row += 1
        print("Total Tubes:", len(tube_set))    
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
    
    
    for city in ['location_phx','location_other','location_sf']:
        tube_set = set()
        dup_labs = np.zeros(len(dups),dtype=int)
        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                    for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                        dupss = road_json['db'][videoname]['frames'][frame]['annos'][anno]['duplex_ids']
                        tube_id =  road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid']
                        for dup in dupss:
                            dup_labs[dup] += 1
                        tube_set.add(tube_id)
        print(city+"  duplex labs: "+ d_path)
        worksheet.write(row, col, city+"  duplex labs: "+ d_path)
        row += 1
        worksheet.write(row, col, "Tubes: ")
        worksheet.write(row, col+1, len(tube_set))
        row += 1   
        worksheet.write(row, col, "Total: ")
        worksheet.write(row, col+1, sum(dup_labs))
        row += 1
        print("Total Tubes:", len(tube_set))  
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
    
    
    for city in ['location_phx','location_other','location_sf']:
        tube_set = set()
        trip_labs = np.zeros(len(trips),dtype=int)
        for videoname in road_json['db']:
            if road_json['db'][videoname]['location'] != city:
                continue
            for frame in road_json['db'][videoname]['frames']:
                if 'annos' in road_json['db'][videoname]['frames'][frame].keys():
                    for anno in road_json['db'][videoname]['frames'][frame]['annos']:
                        tripss = road_json['db'][videoname]['frames'][frame]['annos'][anno]['triplet_ids']
                        tube_id =  road_json['db'][videoname]['frames'][frame]['annos'][anno]['tube_uid']
                        for trip in tripss:
                            trip_labs[trip] += 1
                        tube_set.add(tube_id)
        print(city+"  triplet labs: "+ d_path)
        worksheet.write(row, col, city+"  triplet labs: "+ d_path)
        row += 1
        worksheet.write(row, col, "Tubes: ")
        worksheet.write(row, col+1, len(tube_set))
        row += 1   
        worksheet.write(row, col, "Total: ")
        worksheet.write(row, col+1, sum(trip_labs))
        row += 1  
        print("Total Tubes:", len(tube_set))  
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