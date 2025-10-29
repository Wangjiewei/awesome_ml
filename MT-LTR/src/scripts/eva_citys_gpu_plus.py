#encoding=utf8
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import json

import os
gpu_ids = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

eva_cityid = '55000199'
output_file = ''
input_path = ''
poi_vec_path = ''
query_vec_path = ''

batch_size = int(50)

#
#

#
#

#

#
#

#
#

poiid_list = []
poi_vec_list = []
poi_text_list = []

poi_idx_dic = {}

idx = 0
for line in open(input_path + poi_vec_path):
    items = line.strip().split('\t')
    if len(line) < 4: continue
    #
    (doctype, name_addr, poiid, vec) = line[:4]
    if doctype != 'doc': continue

    vec_list = [float(x) for x in vec.split(',')]
    poiid_list.append(poiid)
    poi_vec_list.append(vec_list)
    poi_text_list.append(name_addr)
    poi_idx_dic[poiid] = (idx, name_addr)
    idx += 1
    if(idx % 10000 == 0):
        print('load poi vec num:', idx)


poi_vec_array = np.array(poi_vec_list)
print()
print()

#


query_vec_list = []
label_poi_list = []
query_list = []
idj = 0
knn_k = 10
neighbor_k = 10

outFile = open(output_file, 'w')
recall_at = {}

sub_positive_case = 0

def cal_case(label_poi_list, query_vec_list, idj):
    print('processed: %d', idj)
    query_vec_array = np.array(query_vec_list)
    distances = cosine_distances(query_vec_array, poi_vec_array)
    neighbors = np.argpartition(distances, range(0,neighbor_k))

    global sub_positive_case
    print(neighbors[:knn_k])

    for j in range(0, len(neighbors)):
        label_poi = label_poi_list[j]
        has_recall = False
        include_case = False
        for i in range(0, knn_k):
            if poiid_list[neighbors[j][i]] == label_poi:
                has_recall = True
                recall_at.setdefault(i, 0)
                recall_at[i] += 1
                print('label_poi: %s %d %d'%(label_poi, i, idj))
                print(recall_at)
                break
            poi_name = poi_text_list[neighbors[j][i]].split('|')[0]
            if query_list[j] in poi_name:
                include_case = True

        case_type = 'positive_case' if has_recall else 'negative_case'
        include_type = 'include_true' if include_case else 'include_false'
        if(case_type == 'positive_case' or include_case == 'include_true'): sub_positive_case += 1

        print()
        outFile.write('%s: query: %s label: %s # %s\n'%(case_type, query_list[j], label_poi, include_type))
        for i in range(0, neighbor_k):
            n_poi_text = poi_text_list[neighbors[j][i]]
            n_poi_id = poiid_list[neighbors[j][i]]
            if(i<knn_k):
                if poiid_list[neighbors[j][i]] == label_poi:
                    has_recall = True
                    print()
                    outFile.write()
                else:
                    print()
                    outFile.write()
        if not has_recall:
            print()
            outFile.write('\n')
        outFile.write()



for line in open(input_path + query_vec_path):
    line = line.strip().split('\t')
    if len(line) < 4: continue
    #
    (doctype, query, label_poi, vec) = line[:4]
    if doctype != 'query': continue

    #
    if label_poi not in poi_idx_dic: continue
    #

    vec_list = [float(x) for x in vec.split(',')]
    label_poi_list.append(label_poi)
    query_vec_list.append(vec_list)
    query_list.append(query)

    idj += 1
    if(idj % batch_size == 0):

        cal_case(label_poi_list, query_vec_list, idj)

        label_poi_list = []
        query_vec_list = []
        query_list = []

    outFile.flush()

if len(query_vec_list) > 0:
    cal_case(label_poi_list, query_vec_list, idj)


total_positive_case = 0
for i in range(0, knn_k):
    if i in recall_at:
        total_positive_case += recall_at[i]
        print()

print()
print()

outFile.close()

print('done')