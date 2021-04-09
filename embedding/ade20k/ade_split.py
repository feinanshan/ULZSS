cat_stuff_in1k = ['wall',   'building', 'sky',   'road',     'grass', 
				  'ground', 'mountain', 'water', 'house',    'sea', 
				  'rug',    'field',    'sand',  'stand',    'river', 
				  'bridge', 'hovel',    'tower', 'track',    'land', 
				  'moving', 'stage',    'belt',  'swimming', 'lake',   'pier']





cat_stuff_notin1k = ['empty', 'flooring', 'ceiling',   'pavement', 'skyscraper', 
					 'path',  'runway',   'staircase', 'hill',     'falls']


cat_obj_in1k = ['tree',      'bed',          'window',     'cabinet', 'door', 
                'table',     'life',         'curtain',    'chair',   'car', 
                'picture',   'couch',        'mirror',     'seat',    'fence',
                'desk',      'rock',         'wardrobe',   'lamp',    'bathtub', 
                'rail',      'base',         'box',        'sign',    'chest', 
                'open',      'refrigerator', 'case',       'pool',    'pillow',  
                'screen',    'bookcase',     'screen',     'coffee',  'toilet', 
                'flower',    'book',         'bench',      'stove',   'tree', 
                'island',    'computer',     'chair',      'boat',    'bar', 
                'machine',   'bus',          'towel',      'light',   'truck', 
                'street',    'stall',        'television', 'plane',   'clothes', 
                'pole',      'bannister',    'puff',       'bottle',  'poster',  
                'van',       'ship',         'fountain',   'washer', 	'toy', 
                'barrel',    'basket',       'tent',       'bag',     'cradle', 
                'oven',      'ball',         'food',       'tank',    'trade', 
                'microwave', 'pot',          'beast',      'bicycle', 'dishwasher', 
                'screen',    'cover',        'vase',       'traffic', 'tray', 
                'ashcan',    'fan',          'screen',     'plate',   'monitor', 
                'board',     'shower',       'radiator',   'glass',   'clock', 
                'flag']




cat_obj_notin1k = [ 'soul',     'shelf',     'armchair', 'cushion',    'pillar', 
					'counter',  'sink',      'steps',    'countertop', 'pendent', 
					'sunblind', 'sideboard', 'canopy',   'stool',      'motorbike', 
					'stair',    'sculpture', 'hood',     'sconce']



cls_stuff_in1k = [ 1,   2,   3,  7,  10, 
                  14,  17,  22,  26, 27, 
                  29,  30,  47,  52, 61, 
				  62,  80,  85,  92, 95, 
				  97,  102, 106, 110, 129, 141]

cls_stuff_notin1k = [  0,  4,  6, 12, 49, 
                      53, 55, 60, 69, 114]

cls_obj_in1k = [  5,   8,   9,  11,  15, 
                 16,  18,  19,  20,  21, 
                 23,  24,  28,  32,  33, 
				 34,  35,  36,  37,  38, 
				 39,  41,  42,  44,  45, 
				 50,  51,  56,  57,  58, 
				 59,  63,  64,  65,  66, 
				 67,  68,  70,  72,  73, 
				 74,  75,  76,  77,  78, 
				 79,  81,  82,  83,  84, 
				 88,  89,  90,  91,  93, 
				 94,  96,  98,  99, 101, 
				103, 104, 105, 108, 109, 
				112, 113, 115, 116, 118, 
				119, 120, 121, 123, 124, 
				125, 126, 127, 128, 130, 
				131, 132, 136, 137, 138, 
				139, 140, 142, 143, 144, 
				145, 146, 147, 148, 149, 
				150]

cls_obj_notin1k = [ 13,  25,  31,  40,  43, 
                    46,  48,  54,  71,  86, 
                    87, 100, 107, 111, 117, 
                   122, 133, 134, 135]

# Before valid encoding
cls_unseen_stuff_in1k = [  2,   7,  14, 22, 27, 
						  30,  85,  92, 95, 106, 
						 110, 129, 141]

cls_unseen_stuff_notin1k = [4, 6, 12, 49, 60]

cls_unseen_obj_in1k = [  5,  16,  19,  20,  21, 
                        32,  34,  35,  36,  39, 
                        42,  44,  45,  51,  56, 
                        59,  63,  64,  65,  72, 
                        76,  77,  78,  81,  83, 
                        84,  88,  93,  94,  99, 
                       101, 105, 113, 116, 121, 
                       124, 125, 126, 127, 128, 
                       130, 137, 138, 139, 140, 
                       144, 145, 148]

cls_unseen_obj_notin1k = [25,  31,  46,  54, 71, 
                          87, 100, 111, 134]


# After valid encoding
cls_unseen_stuff_in1k = [  1,  6, 13, 21,  26, 
                          29, 84, 91, 94, 105, 
                         109, 128, 140]

cls_unseen_stuff_notin1k = [3, 5, 11, 48, 59]

cls_unseen_obj_in1k = [ 4, 15, 18, 19, 20, 
                       31, 33, 34, 35, 38, 
                       41, 43, 44, 50, 55, 
                       58, 62, 63, 64, 71, 
                       75, 76, 77, 80, 82, 
                       83, 87, 92, 93, 98, 
                       100, 104, 112, 115, 120, 
                       123, 124, 125, 126, 127, 
                       129, 136, 137, 138, 139, 
                       143, 144, 147]

cls_unseen_obj_notin1k = [24,  30,  45,  53, 70, 
                          86, 99, 110, 133]


##Verification
import os
import numpy as np
import random 
import inspect, re

def verification():
	def varname(p):
		for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
			m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

	ADE20K_PATH = '/home/ping/Desktop/OnGoing/MkSplit/ADE20K/ade20k-150.txt'
	labels_ade = [content.decode('utf-8').split('\t')[-1] for content in np.genfromtxt(ADE20K_PATH, delimiter=':', dtype=None)]

	verify_cls_list  = [cls_stuff_in1k,cls_stuff_notin1k,cls_obj_in1k,cls_obj_notin1k]
	verify_cat_list = [cat_stuff_in1k,cat_stuff_notin1k,cat_obj_in1k,cat_obj_notin1k]

	for cls_list,cat_list in zip(verify_cls_list,verify_cat_list):
		okay=True
		for cls,cat in zip(cls_list,cat_list):
			if labels_ade[cls].find(cat)==-1:
				print("Error:\t%d\t%s\t%s"%(cls,cat,labels_ade[cls]))
				okay=False
				break
		if okay:
				print("Varify:\t%s(%d)\t%s(%d)"%(varname(cls_list),len(cls_list),varname(cat_list),len(cat_list)))

def random_split():

	def random_sel(lst,ratio):
		return random.sample(lst, int(ratio*len(lst)))

	def varname(p):
		for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
			m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

	cls_unseen_stuff_in1k = random_sel(cls_stuff_in1k, 0.5)
	cls_unseen_stuff_notin1k = random_sel(cls_stuff_notin1k, 0.5)
	cls_unseen_obj_in1k = random_sel(cls_obj_in1k, 0.5)
	cls_unseen_obj_notin1k = random_sel(cls_obj_notin1k, 0.5)

	cls_unseen_stuff_in1k.sort()
	cls_unseen_stuff_notin1k.sort()
	cls_unseen_obj_in1k.sort()
	cls_unseen_obj_notin1k.sort()

	print(varname(cls_unseen_stuff_in1k),'=',cls_unseen_stuff_in1k)
	print(varname(cls_unseen_stuff_notin1k),'=',cls_unseen_stuff_notin1k)
	print(varname(cls_unseen_obj_in1k),'=',cls_unseen_obj_in1k)
	print(varname(cls_unseen_obj_notin1k),'=',cls_unseen_obj_notin1k)



	print(len(cls_unseen_stuff_in1k))
	print(len(cls_unseen_stuff_notin1k))
	print(len(cls_unseen_obj_in1k))
	print(len(cls_unseen_obj_notin1k))

if __name__=="__main__":
	random_split()