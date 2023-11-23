
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
import glob

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

    def __init__(self, args, transform=None,full_test=None):

        self.DATA_PATH = args.DATA_PATH

        self.SEQ_LEN = args.SEQ_LEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        # self.MULIT_SCALE = args.MULIT_SCALE
        self.full_test = full_test
        self.used_labels = {"agent_labels": ["Ped", "Car", "Cyc", "Mobike", "SmalVeh", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL"],
                        "av_action_labels": ["AV-Stop","AV-Mov","AV-TurRht","AV-TurLft","AV-MovRht","AV-MovLft"],
                       "action_labels": ["Red", "Amber", "Green", "MovAway", "MovTow", "Mov", "Rev", "Brake", "Stop", "IncatLft", "IncatRht", "HazLit", "TurLft", "TurRht", "MovRht", "MovLft", "Ovtak", "Wait2X", "XingFmLft", "XingFmRht", "Xing", "PushObj"],
                       "loc_labels": ["VehLane", "OutgoLane", "OutgoCycLane", "OutgoBusLane", "IncomLane", "IncomCycLane", "IncomBusLane", "Pav", "LftPav", "RhtPav", "Jun", "xing", "BusStop", "parking", "LftParking", "rightParking"],
                       "duplex_labels": ["Ped-MovAway", "Ped-MovTow", "Ped-Mov", "Ped-Stop", "Ped-Wait2X", "Ped-XingFmLft", "Ped-XingFmRht", "Ped-Xing", "Ped-PushObj", "Car-MovAway", "Car-MovTow", "Car-Brake", "Car-Stop", "Car-IncatLft", "Car-IncatRht", "Car-HazLit", "Car-TurLft", "Car-TurRht", "Car-MovRht", "Car-MovLft", "Car-XingFmLft", "Car-XingFmRht", "Cyc-MovAway", "Cyc-MovTow", "Cyc-Stop", "Mobike-Stop", "MedVeh-MovAway", "MedVeh-MovTow", "MedVeh-Brake", "MedVeh-Stop", "MedVeh-IncatLft", "MedVeh-IncatRht", "MedVeh-HazLit", "MedVeh-TurRht", "MedVeh-XingFmLft", "MedVeh-XingFmRht", "LarVeh-MovAway", "LarVeh-MovTow", "LarVeh-Stop", "LarVeh-HazLit", "Bus-MovAway", "Bus-MovTow", "Bus-Brake", "Bus-Stop", "Bus-HazLit", "EmVeh-Stop", "TL-Red", "TL-Amber", "TL-Green"], 
                       "triplet_labels": ["Ped-MovAway-LftPav", "Ped-MovAway-RhtPav", "Ped-MovAway-Jun", "Ped-MovTow-LftPav", "Ped-MovTow-RhtPav", "Ped-MovTow-Jun", "Ped-Mov-OutgoLane", "Ped-Mov-Pav", "Ped-Mov-RhtPav", "Ped-Stop-OutgoLane", "Ped-Stop-Pav", "Ped-Stop-LftPav", "Ped-Stop-RhtPav", "Ped-Stop-BusStop", "Ped-Wait2X-RhtPav", "Ped-Wait2X-Jun", "Ped-XingFmLft-Jun", "Ped-XingFmRht-Jun", "Ped-XingFmRht-xing", "Ped-Xing-Jun", "Ped-PushObj-LftPav", "Ped-PushObj-RhtPav", "Car-MovAway-VehLane", "Car-MovAway-OutgoLane", "Car-MovAway-Jun", "Car-MovTow-VehLane", "Car-MovTow-IncomLane", "Car-MovTow-Jun", "Car-Brake-VehLane", "Car-Brake-OutgoLane", "Car-Brake-Jun", "Car-Stop-VehLane", "Car-Stop-OutgoLane", "Car-Stop-IncomLane", "Car-Stop-Jun", "Car-Stop-parking", "Car-IncatLft-VehLane", "Car-IncatLft-OutgoLane", "Car-IncatLft-IncomLane", "Car-IncatLft-Jun", "Car-IncatRht-VehLane", "Car-IncatRht-OutgoLane", "Car-IncatRht-IncomLane", "Car-IncatRht-Jun", "Car-HazLit-IncomLane", "Car-TurLft-VehLane", "Car-TurLft-Jun", "Car-TurRht-Jun", "Car-MovRht-OutgoLane", "Car-MovLft-VehLane", "Car-MovLft-OutgoLane", "Car-XingFmLft-Jun", "Car-XingFmRht-Jun", "Cyc-MovAway-OutgoCycLane", "Cyc-MovAway-RhtPav", "Cyc-MovTow-IncomLane", "Cyc-MovTow-RhtPav", "MedVeh-MovAway-VehLane", "MedVeh-MovAway-OutgoLane", "MedVeh-MovAway-Jun", "MedVeh-MovTow-IncomLane", "MedVeh-MovTow-Jun", "MedVeh-Brake-VehLane", "MedVeh-Brake-OutgoLane", "MedVeh-Brake-Jun", "MedVeh-Stop-VehLane", "MedVeh-Stop-OutgoLane", "MedVeh-Stop-IncomLane", "MedVeh-Stop-Jun", "MedVeh-Stop-parking", "MedVeh-IncatLft-IncomLane", "MedVeh-IncatRht-Jun", "MedVeh-TurRht-Jun", "MedVeh-XingFmLft-Jun", "MedVeh-XingFmRht-Jun", "LarVeh-MovAway-VehLane", "LarVeh-MovTow-IncomLane", "LarVeh-Stop-VehLane", "LarVeh-Stop-Jun", "Bus-MovAway-OutgoLane", "Bus-MovTow-IncomLane", "Bus-Stop-VehLane", "Bus-Stop-OutgoLane", "Bus-Stop-IncomLane", "Bus-Stop-Jun", "Bus-HazLit-OutgoLane"]}
        
        self.transform = transform
        self.ids = list()
        self._make_lists()
        self.num_label_types = len(self.label_types)

    def _make_lists(self):
        # loading just for label types etc...
        self.anno_file  = os.path.join("../road_waymo", 'road_waymo_trainval_v1.0.json')
        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
       
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

        file_list = [x for x in sorted([x for x in os.listdir(self.DATA_PATH+'/')])]

        for i in range (0,len(file_list),self.SEQ_LEN):
            seq_list = []
            for j in range(self.SEQ_LEN):
                seq_list.append(file_list[i+j])
            self.ids.append(seq_list)

        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = self.used_labels[name+'_labels']
            self.all_classes.append(labels)
        self.label_types = ['agent_ness'] + self.label_types

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        imgs_list = self.ids[index]
        images = []
        img_names = []
        for i in range(self.SEQ_LEN):
            img_name = self.DATA_PATH+'/'+imgs_list[i]
            img_names.append(img_name)
            img = Image.open(img_name).convert('RGB')
            images.append(img)
        clip = self.transform(images)

        return clip, img_names
