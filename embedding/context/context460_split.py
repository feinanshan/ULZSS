
cat_seen = ['aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 
			'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 
			'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 
			'cow', 'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 
			'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 'motorbike', 
			'mountain', 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road', 
			'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 
			'table', 'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water', 
			'window', 'wood']

cat_unseen = ['ashtray', 'babycarriage', 'ball', 'balloon', 'basket', 'bathtub', 'birdcage', 
			  'birdfeeder', 'birdnest', 'board', 'bone', 'bowl', 'box', 'brick', 'bridge', 
			  'brush', 'bucket', 'cage', 'cake', 'calendar', 'camera', 'can', 'candle', 
			  'candleholder', 'cap', 'card', 'cart', 'case', 'cd', 'cellphone', 'cello', 
			  'chopstick', 'clock', 'concrete', 'cone', 'container', 'controller', 'counter', 
			  'cushion', 'disccase', 'drum', 'drumkit', 'duck', 'egg', 'equipment', 'exhibitionbooth', 
			  'eyeglass', 'fan', 'fireplace', 'fish', 'fishtank', 'flag', 'flagstaff', 'fork', 'frame', 
			  'fridge', 'gamemachine', 'glass', 'glove', 'goal', 'grandstand', 'guardrail', 'guitar', 
			  'handcart', 'handle', 'handrail', 'hat', 'helmet', 'horse-drawncarriage', 'ice', 'ipod', 
			  'jar', 'key', 'kitchenrange', 'knife', 'ladder', 'laptop', 'mat', 'menu', 'metal', 'microphone', 
			  'microwave', 'mirror', 'mousepad', 'musicalinstrument', 'napkin', 'newspaper', 'oar', 'outlet', 
			  'oven', 'pack', 'pan', 'paper', 'papercutter', 'parasol', 'pen', 'pencontainer', 'piano', 'picture', 
			  'pillar', 'pillow', 'pipe', 'plant', 'plastic', 'player', 'pole', 'poster', 'pot', 'printer', 'pumpkin', 
			  'radiator', 'rail', 'rangehood', 'receiver', 'remotecontrol', 'rope', 'rug', 'saddle', 'sand', 'sculpture', 
			  'shoe', 'signallight', 'sink', 'ski', 'sled', 'smoke', 'speaker', 'spicecontainer', 'spoon', 'stage', 'stair',
			  'stool', 'stove', 'tap', 'tarp', 'telephone', 'tent', 'tire', 'tool', 'towel', 'toy', 'trashbin', 'tray', 'umbrella',
			   'unknown', 'vacuumcleaner', 'videocamera', 'videogameconsole', 'videoplayer', 'videotape', 'violin', 'washingmachine', 
			   'waterdispenser', 'wheel', 'wineglass', 'wire']

cls_seen = [2, 9, 18, 19, 22, 23, 25, 31, 33, 34, 44, 45, 46, 59, 65, 68, 72, 
			80, 85, 98, 104, 105, 113, 115, 144, 158, 159, 162, 187, 189, 207, 220, 
			232, 258, 259, 260, 284, 295, 296, 308, 324, 326, 347, 349, 354, 355, 360, 
			366, 368, 397, 415, 416, 420, 424, 427, 440, 445, 454, 458]

cls_unseen = [6, 8, 10, 11, 15, 17, 26, 27, 28, 30, 32, 36, 37, 39, 40, 42, 43, 48,
			 49, 51, 53, 55, 56, 57, 58, 60, 61, 62, 66, 69, 70, 75, 78, 86, 87, 88,
			 90, 96, 106, 110, 122, 123, 124, 128, 136, 138, 140, 141, 148, 149, 150,
			 154, 155, 165, 169, 170, 176, 181, 184, 185, 186, 190, 191, 194, 195, 196,
			 199, 204, 208, 211, 213, 216, 219, 221, 223, 225, 228, 244, 247, 248, 250,
			 251, 252, 261, 262, 263, 265, 266, 268, 269, 271, 272, 273, 275, 277, 281,
			 282, 286, 287, 289, 290, 291, 293, 294, 297, 303, 306, 307, 309, 311, 314,
			 316, 319, 320, 323, 329, 330, 333, 334, 342, 350, 356, 357, 359, 361, 363,
			 371, 373, 374, 377, 378, 383, 384, 400, 402, 403, 405, 406, 410, 412, 413,
			 418, 419, 430, 431, 432, 434, 435, 436, 437, 438, 443, 446, 452, 456, 457]


cat_seen_notin1k = ['aeroplane', 'bedclothes', 'ceiling', 'floor', 'motorbike', 'person', 
					'pottedplant', 'shelves', 'sidewalk', 'sofa', 'tvmonitor']

cat_seen_in1k = ['bag', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus', 
					'cabinet', 'car', 'cat', 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog', 
					'door', 'fence', 'flower', 'food', 'grass', 'ground', 'horse', 'keyboard', 'light', 
					'mountain', 'mouse', 'plate', 'platform', 'road', 'rock', 'sheep', 'sign', 'sky', 
					'snow', 'table', 'track', 'train', 'tree', 'truck', 'wall', 'water', 'window', 'wood']

cat_unseen_notin1k = ['ashtray', 'babycarriage', 'birdcage', 'birdfeeder', 'birdnest', 'brick', 'cage', 'cake', 
					  'calendar', 'candleholder', 'cd', 'chopstick', 'concrete', 'cone', 'controller', 'counter', 
					  'cushion', 'disccase', 'drumkit', 'equipment', 'exhibitionbooth', 'eyeglass', 'fireplace', 
					  'fishtank', 'frame', 'fridge', 'gamemachine', 'glove', 'goal', 'grandstand', 'guardrail', 
					  'handcart', 'handle', 'horse-drawncarriage', 'jar', 'kitchenrange', 'ladder', 'metal', 
					  'mousepad', 'musicalinstrument', 'newspaper', 'outlet', 'papercutter', 'parasol', 
					  'pencontainer', 'pillar', 'plant', 'rangehood', 'receiver', 'remotecontrol', 'saddle', 
					  'sculpture', 'signallight', 'sink', 'smoke', 'spicecontainer', 'stair', 'stool', 'tarp', 
					  'tire', 'trashbin', 'unknown', 'vacuumcleaner', 'videocamera', 'videogameconsole', 
					  'videoplayer', 'videotape', 'washingmachine', 'waterdispenser', 'wineglass']

cat_unseen_in1k = ['ball', 'balloon', 'basket', 'bathtub', 'board', 'bone', 'bowl', 'box', 'bridge', 
					'brush', 'bucket', 'camera', 'can', 'candle', 'cap', 'card', 'cart', 'case', 
					'cellphone', 'cello', 'clock', 'container', 'drum', 'duck', 'egg', 'fan', 
					'fish', 'flag', 'flagstaff', 'fork', 'glass', 'guitar', 'handrail', 'hat', 
					'helmet', 'ice', 'ipod', 'key', 'knife', 'laptop', 'mat', 'menu', 'microphone', 
					'microwave', 'mirror', 'napkin', 'oar', 'oven', 'pack', 'pan', 'paper', 'pen', 
					'piano', 'picture', 'pillow', 'pipe', 'plastic', 'player', 'pole', 'poster', 
					'pot', 'printer', 'pumpkin', 'radiator', 'rail', 'rope', 'rug', 'sand', 'shoe', 
					'ski', 'sled', 'speaker', 'spoon', 'stage', 'stove', 'tap', 'telephone', 'tent', 
					'tool', 'towel', 'toy', 'tray', 'umbrella', 'violin', 'wheel', 'wire']

cls_seen_notin1k = [2, 19, 68, 158, 258, 284, 308, 349, 354, 368, 427]

cls_seen_in1k = [9, 18, 22, 23, 25, 31, 33, 34, 44, 45, 46, 59, 65, 72, 80, 85, 
				98, 104, 105, 113, 115, 144, 159, 162, 187, 189, 207, 220, 232, 
				259, 260, 295, 296, 324, 326, 347, 355, 360, 366, 397, 415, 416, 
				420, 424, 440, 445, 454, 458]

#Before encoding
cls_unseen_notin1k = [6, 8, 26, 27, 28, 39, 48, 49, 51, 57, 66, 75, 86, 87, 90, 96,
						106, 110, 123, 136, 138, 140, 148, 150, 169, 170, 176, 
						184, 185, 186, 190, 194, 195, 208, 216, 221, 225, 248, 
						261, 262, 265, 268, 275, 277, 282, 289, 293, 319, 320, 
						323, 333, 342, 356, 357, 363, 373, 378, 383, 402, 406, 
						418, 431, 432, 434, 435, 436, 437, 443, 446, 456]

cls_unseen_in1k = [ 10, 11, 15, 17, 30, 32, 36, 37, 40, 42, 43, 53, 55, 
					56, 58, 60, 61, 62, 69, 70, 78, 88, 122, 124, 128, 
					141, 149, 154, 155, 165, 181, 191, 196, 199, 204, 
					211, 213, 219, 223, 228, 244, 247, 250, 251, 252, 
					263, 266, 269, 271, 272, 273, 281, 286, 287, 290, 
					291, 294, 297, 303, 306, 307, 309, 311, 314, 316, 
					329, 330, 334, 350, 359, 361, 371, 374, 377, 384, 
					400, 403, 405, 410, 412, 413, 419, 430, 438, 452, 457]

#After encoding

cls_unseen_notin1k = [1, 2, 13, 14, 15, 23, 30, 31, 32, 36, 43, 48, 52, 
					 53, 55, 56, 60, 61, 65, 68, 69, 70, 73, 75, 82, 83, 
					 84, 86, 87, 88, 91, 93, 94, 99, 102, 105, 107, 112, 
					 119, 120, 122, 124, 129, 130, 132, 136, 139, 152, 153, 
					 154, 159, 161, 167, 168, 172, 176, 179, 180, 184, 187, 
					 193, 199, 200, 201, 202, 203, 204, 207, 209, 212]

cls_unseen_in1k = [4, 5, 6, 7, 16, 18, 21, 22, 24, 25, 26, 33, 34, 35, 
					37, 39, 40, 41, 45, 46, 49, 54, 64, 66, 67, 71, 74, 
					76, 77, 81, 85, 92, 95, 96, 97, 100, 101, 103, 106, 
					108, 110, 111, 113, 114, 115, 121, 123, 125, 126, 127, 
					128, 131, 134, 135, 137, 138, 140, 143, 144, 145, 146, 
					148, 149, 150, 151, 157, 158, 160, 164, 169, 171, 175, 
					177, 178, 181, 183, 185, 186, 188, 189, 190, 194, 198, 
					205, 210, 213]

cls_unseen = [1, 2, 4, 5, 6, 7, 13, 14, 15, 16, 18, 
			  21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 
			  34, 35, 36, 37, 39, 40, 41, 43, 45, 46, 
			  48, 49, 52, 53, 54, 55, 56, 60, 61, 64, 
			  65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 
			  76, 77, 81, 82, 83, 84, 85, 86, 87, 88, 
			  91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 
			  102, 103, 105, 106, 107, 108, 110, 111, 112, 
			  113, 114, 115, 119, 120, 121, 122, 123, 124, 
			  125, 126, 127, 128, 129, 130, 131, 132, 134, 
			  135, 136, 137, 138, 139, 140, 143, 144, 145, 
			  146, 148, 149, 150, 151, 152, 153, 154, 157, 
			  158, 159, 160, 161, 164, 167, 168, 169, 171, 
			  172, 175, 176, 177, 178, 179, 180, 181, 183, 
			  184, 185, 186, 187, 188, 189, 190, 193, 194, 
			  198, 199, 200, 201, 202, 203, 204, 205, 207, 
			  209, 210, 212, 213]




##Verification
import os
import numpy as np

import inspect, re

def verification():
	def varname(p):
		for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
			m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

	CAT460_PATH = '/home/ping/Desktop/OnGoing/MkSplit/PascalContext/context460.txt'
	labels_460 = [label.decode('utf-8').replace(' ','') for idx, label in np.genfromtxt(CAT460_PATH, delimiter=':', dtype=None)]

	verify_cls_list  = [cls_seen_notin1k,cls_seen_in1k,cls_unseen_notin1k,cls_unseen_in1k]
	verify_cat_list = [cat_seen_notin1k,cat_seen_in1k,cat_unseen_notin1k,cat_unseen_in1k]

	for cls_list,cat_list in zip(verify_cls_list,verify_cat_list):
		okay=True
		for cls,cat in zip(cls_list,cat_list):
			if labels_460[cls].find(cat)==-1:
				print("Error:\t%d\t%s\t%s"%(cls,cat,labels_460[cls]))
				okay=False
				break
		if okay:
				print("Varify:\t%s(%d)\t%s(%d)"%(varname(cls_list),len(cls_list),varname(cat_list),len(cat_list)))

if __name__=="__main__":
	verification()