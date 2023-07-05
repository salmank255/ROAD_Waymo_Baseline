
"""

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""
import cv2
import json, os
import torch
import pdb, time
import torch.utils as tutils
import pickle
from .transforms import get_clip_list_resized
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES =   True
from PIL import Image, ImageDraw
from modules.tube_helper import make_gt_tube
import random as random
from modules import utils 
from random import shuffle

logger = utils.get_logger(__name__)

g_w, g_h = 0, 0






def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids


def get_gt_video_list(anno_file, SUBSETS):
    """Get video list form ground truth videos used in subset 
    and their ground truth tubes """

    with open(anno_file, 'r') as fff:
        final_annots = json.load(fff)

    video_list = []
    for videoname in final_annots['db']:
        if is_part_of_subsets(final_annots['db'][videoname]['split_ids'], SUBSETS):
            video_list.append(videoname)

    return video_list


def get_filtered_tubes(label_key, final_annots, videoname):
    if not label_key in final_annots['db'][videoname]:
        return []
    key_tubes = final_annots['db'][videoname][label_key]
    all_labels = final_annots['all_'+label_key.replace('tubes','labels')]
    labels = final_annots[label_key.replace('tubes','labels')]
    filtered_tubes = []

    for _ , tube in key_tubes.items():
        label_id = tube['label_id']
        label = all_labels[label_id]
        if label in labels:
            new_label_id = labels.index(label)
            # temp_tube = GtTube(new_label_id)
            frames = []
            boxes = []
            if 'annos' in tube.keys():
                for fn, anno_id in tube['annos'].items():
                    frames.append(int(fn))
                    anno = final_annots['db'][videoname]['frames'][fn]['annos'][anno_id]
                    box = anno['box'].copy()
                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1.0, max(0, box[bi]))
                        box[bi] = box[bi]*g_h if bi % 2 == 0 else box[bi]*g_w
                    boxes.append(box)
            else:
                for fn in tube['frames']:
                    frames.append(int(fn))

            temp_tube = make_gt_tube(frames, boxes, new_label_id)
            filtered_tubes.append(temp_tube)
            
    return filtered_tubes


def get_filtered_frames(label_key, final_annots, videoname, filtered_gts):
    
    frames = final_annots['db'][videoname]['frames']
    if label_key == 'agent_ness':
        all_labels = []
        labels = []
    else:
        all_labels = final_annots['all_'+label_key+'_labels']
        labels = final_annots[label_key+'_labels']
    
    for frame_id , frame in frames.items():
        frame_name = '{:05d}'.format(int(frame_id))
        if frame['annotated']>0:
            all_boxes = []
            if 'annos' in frame:
                frame_annos = frame['annos']
                for key in frame_annos:
                    anno = frame_annos[key]
                    box = np.asarray(anno['box'].copy())
                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1.0, max(0, box[bi]))
                        box[bi] = box[bi]*g_h if bi % 2 == 0 else box[bi]*g_w
                    if label_key == 'agent_ness':
                        filtered_ids = [0]
                    else:
                        filtered_ids = filter_labels(anno[label_key+'_ids'], all_labels, labels)

                    if len(filtered_ids)>0:
                        all_boxes.append([box, filtered_ids])
                
            filtered_gts[videoname+frame_name] = all_boxes
            
    return filtered_gts

def get_av_actions(final_annots, videoname):
    label_key = 'av_action'
    frames = final_annots['db'][videoname]['frames']
    all_labels = final_annots['all_'+label_key+'_labels']
    labels = final_annots[label_key+'_labels']
    
    filtered_gts = {}
    for frame_id , frame in frames.items():
        frame_name = '{:05d}'.format(int(frame_id))
        if frame['annotated']>0:
            gts = filter_labels(frame[label_key+'_ids'], all_labels, labels)
            filtered_gts[videoname+frame_name] = gts
            
    return filtered_gts

def get_video_tubes(final_annots, videoname):
    
    tubes = {}
    for key in final_annots['db'][videoname].keys():
        if key.endswith('tubes'):
            filtered_tubes = get_filtered_tubes(key, final_annots, videoname)
            tubes[key] = filtered_tubes
    
    return tubes


def is_part_of_subsets(split_ids, SUBSETS):
    
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    
    return is_it


class VideoDataset(tutils.data.Dataset):
    """
    ROAD Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, dataset, train=True, input_type='rgb', transform=None, 
                skip_step=1, full_test=False):

        self.ANCHOR_TYPE =  args.ANCHOR_TYPE 
        self.DATASET = dataset
        self.MODE = args.MODE
        self.TEST_SUBSETS = args.TEST_SUBSETS  
        if train == True:
            if self.DATASET == 'road':
                self.SUBSETS = ['train_3']
            elif self.DATASET == 'road_waymo':
                self.SUBSETS = ['train']
        elif self.MODE == 'Train':
            if self.DATASET == 'road':
                self.SUBSETS = ['val_3']
            elif self.DATASET == 'road_waymo':
                self.SUBSETS = ['val']
        elif self.TEST_SUBSETS == ['val']:
            if self.DATASET == 'road':
                self.SUBSETS = ['val_3']
            elif self.DATASET == 'road_waymo':
                self.SUBSETS = ['val']
        elif self.TEST_SUBSETS == ['test']:
            if self.DATASET == 'road':
                self.SUBSETS = ['test']
            elif self.DATASET == 'road_waymo':
                self.SUBSETS = ['test']
    
        print(self.SUBSETS)

        self.SEQ_LEN = args.SEQ_LEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        # self.MULIT_SCALE = args.MULIT_SCALE
        self.full_test = full_test
        self.skip_step = skip_step #max(skip_step, self.SEQ_LEN*self.MIN_SEQ_STEP/2)
        self.num_steps = max(1, int(self.MAX_SEQ_STEP - self.MIN_SEQ_STEP + 1 )//2)
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.DATA_ROOT + self.DATASET + '/'
        self._imgpath = os.path.join(self.root, self.input_type)
        self.anno_root = self.root
        # if len(args.ANNO_ROOT)>1:
        #     self.anno_root = args.ANNO_ROOT 
        self.used_labels = {"agent_labels": ["Ped", "Car", "Cyc", "Mobike", "SmalVeh", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL"],
                        "av_action_labels": ["AV-Stop","AV-Mov","AV-TurRht","AV-TurLft","AV-MovRht","AV-MovLft"],
                       "action_labels": ["Red", "Amber", "Green", "MovAway", "MovTow", "Mov", "Rev", "Brake", "Stop", "IncatLft", "IncatRht", "HazLit", "TurLft", "TurRht", "MovRht", "MovLft", "Ovtak", "Wait2X", "XingFmLft", "XingFmRht", "Xing", "PushObj"],
                       "loc_labels": ["VehLane", "OutgoLane", "OutgoCycLane", "OutgoBusLane", "IncomLane", "IncomCycLane", "IncomBusLane", "Pav", "LftPav", "RhtPav", "Jun", "xing", "BusStop", "parking", "LftParking", "rightParking"],
                       "duplex_labels": ["Ped-MovAway", "Ped-MovTow", "Ped-Mov", "Ped-Stop", "Ped-Wait2X", "Ped-XingFmLft", "Ped-XingFmRht", "Ped-Xing", "Ped-PushObj", "Car-MovAway", "Car-MovTow", "Car-Brake", "Car-Stop", "Car-IncatLft", "Car-IncatRht", "Car-HazLit", "Car-TurLft", "Car-TurRht", "Car-MovRht", "Car-MovLft", "Car-XingFmLft", "Car-XingFmRht", "Cyc-MovAway", "Cyc-MovTow", "Cyc-Stop", "Mobike-Stop", "MedVeh-MovAway", "MedVeh-MovTow", "MedVeh-Brake", "MedVeh-Stop", "MedVeh-IncatLft", "MedVeh-IncatRht", "MedVeh-HazLit", "MedVeh-TurRht", "MedVeh-XingFmLft", "MedVeh-XingFmRht", "LarVeh-MovAway", "LarVeh-MovTow", "LarVeh-Stop", "LarVeh-HazLit", "Bus-MovAway", "Bus-MovTow", "Bus-Brake", "Bus-Stop", "Bus-HazLit", "EmVeh-Stop", "TL-Red", "TL-Amber", "TL-Green"], 
                       "triplet_labels": ["Ped-MovAway-LftPav", "Ped-MovAway-RhtPav", "Ped-MovAway-Jun", "Ped-MovTow-LftPav", "Ped-MovTow-RhtPav", "Ped-MovTow-Jun", "Ped-Mov-OutgoLane", "Ped-Mov-Pav", "Ped-Mov-RhtPav", "Ped-Stop-OutgoLane", "Ped-Stop-Pav", "Ped-Stop-LftPav", "Ped-Stop-RhtPav", "Ped-Stop-BusStop", "Ped-Wait2X-RhtPav", "Ped-Wait2X-Jun", "Ped-XingFmLft-Jun", "Ped-XingFmRht-Jun", "Ped-XingFmRht-xing", "Ped-Xing-Jun", "Ped-PushObj-LftPav", "Ped-PushObj-RhtPav", "Car-MovAway-VehLane", "Car-MovAway-OutgoLane", "Car-MovAway-Jun", "Car-MovTow-VehLane", "Car-MovTow-IncomLane", "Car-MovTow-Jun", "Car-Brake-VehLane", "Car-Brake-OutgoLane", "Car-Brake-Jun", "Car-Stop-VehLane", "Car-Stop-OutgoLane", "Car-Stop-IncomLane", "Car-Stop-Jun", "Car-Stop-parking", "Car-IncatLft-VehLane", "Car-IncatLft-OutgoLane", "Car-IncatLft-IncomLane", "Car-IncatLft-Jun", "Car-IncatRht-VehLane", "Car-IncatRht-OutgoLane", "Car-IncatRht-IncomLane", "Car-IncatRht-Jun", "Car-HazLit-IncomLane", "Car-TurLft-VehLane", "Car-TurLft-Jun", "Car-TurRht-Jun", "Car-MovRht-OutgoLane", "Car-MovLft-VehLane", "Car-MovLft-OutgoLane", "Car-XingFmLft-Jun", "Car-XingFmRht-Jun", "Cyc-MovAway-OutgoCycLane", "Cyc-MovAway-RhtPav", "Cyc-MovTow-IncomLane", "Cyc-MovTow-RhtPav", "MedVeh-MovAway-VehLane", "MedVeh-MovAway-OutgoLane", "MedVeh-MovAway-Jun", "MedVeh-MovTow-IncomLane", "MedVeh-MovTow-Jun", "MedVeh-Brake-VehLane", "MedVeh-Brake-OutgoLane", "MedVeh-Brake-Jun", "MedVeh-Stop-VehLane", "MedVeh-Stop-OutgoLane", "MedVeh-Stop-IncomLane", "MedVeh-Stop-Jun", "MedVeh-Stop-parking", "MedVeh-IncatLft-IncomLane", "MedVeh-IncatRht-Jun", "MedVeh-TurRht-Jun", "MedVeh-XingFmLft-Jun", "MedVeh-XingFmRht-Jun", "LarVeh-MovAway-VehLane", "LarVeh-MovTow-IncomLane", "LarVeh-Stop-VehLane", "LarVeh-Stop-Jun", "Bus-MovAway-OutgoLane", "Bus-MovTow-IncomLane", "Bus-Stop-VehLane", "Bus-Stop-OutgoLane", "Bus-Stop-IncomLane", "Bus-Stop-Jun", "Bus-HazLit-OutgoLane"]}
        
        # self.image_sets = image_sets
        self.transform = transform
        self.ids = list()
        print(self.DATASET)
        if self.DATASET == 'road':
            self._make_lists_road()  
        elif self.DATASET == 'ucf24':
            self._make_lists_ucf24() 
        elif self.DATASET == 'ava':
            self._make_lists_ava() 
        elif self.DATASET == 'road_waymo':
            self._make_lists_road_waymo() 

        else:
            raise Exception('Specfiy corect dataset')
        
        self.num_label_types = len(self.label_types)



        

    def _make_lists_road_waymo(self):
        
        if self.MODE =='train' or self.TEST_SUBSETS == ['val']:
            self.anno_file  = os.path.join(self.root, 'road_waymo_trainval_v1.0.json')
        else:
            self.anno_file  = os.path.join(self.root, 'road_waymo_test_v1.0.json')
        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        # self.label_types = ['agent', 'action', 'loc'] #
        # print(self.label_types)
        # print(rr)

        num_label_type = len(self.label_types)
        self.num_classes = 1 ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            logger.info('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(self.used_labels[name+'_labels'])))
            numc = len(self.used_labels[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = self.used_labels['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)

        counts = np.zeros((len(self.used_labels[self.label_types[-1] + '_labels']), num_label_type), dtype=np.int32)
        # counts = np.zeros((len(final_annots[self.label_types[0] + '_labels']) + len(final_annots[self.label_types[1] + '_labels']) +len(final_annots[self.label_types[2] + '_labels'])  , num_label_type), dtype=np.int32)


        self.video_list = []
        self.numf_list = []
        frame_level_list = []

        # vidnames = sorted(database.keys())
        # vidnames = vidnames[:650]
        # for videoname in vidnames:
        for videoname in sorted(database.keys()):
            # print(is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS))
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            # print(numf)
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    if len(frames[frame_id]['av_action_ids']) == 0:
                        frame_level_annos[frame_index]['ego_label'] = 0
                    elif frames[frame_id]['av_action_ids'][0] >5:
                        frame_level_annos[frame_index]['ego_label'] = 0
                    else:
                        frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]
                    
                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    # temp_img = cv2.imread('../roadpp/rgb-images/'+videoname+'/{:05d}.jpg'.format(frame_num))
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], str(box)+videoname+str(frame_num)
                        assert width==1920 and height==1280, (width, height, box) # for ROAD ++
                        
                        # temp_img = cv2.rectangle(temp_img, (int(box[0]*1920),int(box[1]*1280)), (int(box[2]*1920),int(box[3]*1280)), (255,0,0), 2)
                        # cv2.imwrite('temp_img.png',temp_img)
                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            # print(idx,name)
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], self.used_labels[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc] = 1
                                box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 
                    # print(videoname,frame_num)
                    # print(rr)
                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes

            logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  

            ## make ids
            start_frames = [f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, 1,  -self.skip_step)]
            if self.full_test and 1 not in start_frames:
                start_frames.append(1)
            logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    self.ids.append([video_id, frame_num ,step_list[s]])
        # pdb.set_trace()
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = self.used_labels[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))

        self.label_types = ['agent_ness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr


    def _make_lists_road(self):
        if self.MODE == 'train':
            self.anno_file  = os.path.join(self.root, 'road_trainval_v1.0.json')
        else:
            self.anno_file  = os.path.join(self.root, 'road_test_v1.0.json')
        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        self.label_types =  final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #
        
        num_label_type = len(self.label_types)
        self.num_classes = 1 ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            logger.info('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(self.used_labels[name+'_labels'])))
            numc = len(self.used_labels[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        self.ego_classes = self.used_labels['av_action_labels']
        self.num_ego_classes = len(self.ego_classes)
        
        counts = np.zeros((len(self.used_labels[self.label_types[-1] + '_labels']), num_label_type), dtype=np.int32)

        self.video_list = []
        self.numf_list = []
        frame_level_list = []
        
        # vidnames = sorted(database.keys())
        # vidnames = vidnames[:2]
        # for videoname in vidnames:
        for videoname in sorted(database.keys()):
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    if len(frames[frame_id]['av_action_ids']) == 0:
                        frame_level_annos[frame_index]['ego_label'] = 0
                    elif frames[frame_id]['av_action_ids'][0] >5:
                        frame_level_annos[frame_index]['ego_label'] = 0
                    else:
                        frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]

                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], box
                        assert width==1280 and height==960, (width, height, box) # for ROAD

                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], self.used_labels[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc] = 1
                                box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 

                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes

            logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  

            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, 1,  -self.skip_step)]
            if self.full_test and 1 not in start_frames:
                start_frames.append(1)
            logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    self.ids.append([video_id, frame_num ,step_list[s]])
        # pdb.set_trace()
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = self.used_labels[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))

        self.label_types = ['agent_ness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_info = self.ids[index]
        if self.DATASET == 'ava':
            video_id, start_frame, step_size, keyframe = id_info
        else:
            video_id, start_frame, step_size = id_info
        videoname = self.video_list[video_id]
        images = []
        frame_num = start_frame
        ego_labels = np.zeros(self.SEQ_LEN)-1
        all_boxes = []
        labels = []
        ego_labels = []
        mask = np.zeros(self.SEQ_LEN, dtype=int)
        indexs = []
        img_names = []
        for i in range(self.SEQ_LEN):
            indexs.append(frame_num)
            if self.DATASET != 'ava':
                img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num)
                # img_name = self._imgpath + '/{:s}/img_{:05d}.jpg'.format(videoname, frame_num)
            elif self.DATASET == 'ava':
                img_name = self._imgpath + '/{:s}/{:s}_{:06d}.jpg'.format(videoname, videoname, frame_num)
            img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num)
            img_names.append(img_name)

            img = Image.open(img_name).convert('RGB')
            images.append(img)
            if self.frame_level_list[video_id][frame_num]['labeled']:
                mask[i] = 1
                all_boxes.append(self.frame_level_list[video_id][frame_num]['boxes'].copy())
                labels.append(self.frame_level_list[video_id][frame_num]['labels'].copy())
                ego_labels.append(self.frame_level_list[video_id][frame_num]['ego_label'])
            else:
                all_boxes.append(np.asarray([]))
                labels.append(np.asarray([]))
                ego_labels.append(-1)            
            frame_num += step_size
        
        if self.DATASET == 'ava':
            assert keyframe in indexs, ' keyframe is not in frame {} from startframe {} and stepsize of {}'.format(keyframe, start_frame, step_size)
        clip = self.transform(images)
        height, width = clip.shape[-2:]
        wh = [height, width]
        global g_w, g_h

        g_w, g_h = height, width
        # print('image', wh)
        # print(rr)
        if self.ANCHOR_TYPE == 'RETINA':
            for bb, boxes in enumerate(all_boxes):
                if boxes.shape[0]>0:
                    if boxes[0,0]>1:
                        print(bb, videoname)
                        pdb.set_trace()
                    boxes[:, 0] *= width # width x1
                    boxes[:, 2] *= width # width x2
                    boxes[:, 1] *= height # height y1
                    boxes[:, 3] *= height # height y2

        return clip, all_boxes, labels, ego_labels, index, wh, self.num_classes,videoname,start_frame,img_names,g_w, g_h


def custum_collate(batch):
    
    images = []
    boxes = []
    targets = []
    ego_targets = []
    image_ids = []
    whs = []
    videonames = []
    start_frames = []
    img_namess = []
    g_ws = []
    g_hs = []

    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        targets.append(sample[2])
        ego_targets.append(torch.LongTensor(sample[3]))
        image_ids.append(sample[4])
        whs.append(torch.LongTensor(sample[5]))
        num_classes = sample[6]
        videonames.append(sample[7])
        start_frames.append(sample[8])
        img_namess.append(sample[9])

        g_ws.append(sample[10])
        g_hs.append(sample[11])
        
    counts = []
    max_len = -1
    seq_len = len(boxes[0])
    for bs_ in boxes:
        temp_counts = []
        for bs in bs_:
            max_len = max(max_len, bs.shape[0])
            temp_counts.append(bs.shape[0])
        assert seq_len == len(temp_counts)
        counts.append(temp_counts)
    counts = np.asarray(counts, dtype=np.int_)
    new_boxes = torch.zeros(len(boxes), seq_len, max_len, 4)
    new_targets = torch.zeros([len(boxes), seq_len, max_len, num_classes])
    for c1, bs_ in enumerate(boxes):
        for c2, bs in enumerate(bs_):
            if counts[c1,c2]>0:
                assert bs.shape[0]>0, 'bs'+str(bs)
                new_boxes[c1, c2, :counts[c1,c2], :] = torch.from_numpy(bs)
                targets_temp = targets[c1][c2]
                assert targets_temp.shape[0] == bs.shape[0], 'num of labels and boxes should be same'
                new_targets[c1, c2, :counts[c1,c2], :] = torch.from_numpy(targets_temp)

    # images = torch.stack(images, 0)
    images = get_clip_list_resized(images)
    # print(images.shape)
    return images, new_boxes, new_targets, torch.stack(ego_targets,0), torch.LongTensor(counts), image_ids, torch.stack(whs,0),videonames,start_frames,img_namess,g_ws,g_hs
