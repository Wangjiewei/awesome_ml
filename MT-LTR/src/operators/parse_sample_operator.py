# encoding=utf8
from ast import parse
import re
import sys
import os
import time
import json
import random
import datetime
from pyspark.sql import SparkSession
from datetime import timedelta
from collections import defaultdict
from utils import Utils
from config import Config
import copy
reload(sys)
sys.setdefaultencoding('utf-8')

# 非法字符校验







# 默认解析字段算子







# 解析ww算子














# 解析query成分算子















# 解析 poi name address 算子




















# 生成用户的query geohash





#生成用户的poi geohash





# 解析 da结果的算子























# 解析gbdt特征算子
def get_gbdt_feat(dump_feats, dump_feature_cols):
    cur_libSVM_format_feat = []
    # 非空特征
    if dump_feats != 'null' and dump_feats != 'nullLtrFeatures':
        raw_dump_feats = dump_feats.strip(';').split(';')
        # libSVM格式
        if ":" in dump_feats:
            dump_feats_slice = {}
            for idx, feat in enumerate(raw_dump_feats[:dump_feature_cols]):
                feat_kv = feat.split(':')
                feat_idx = int(feat_kv[0]) - 1
                feat_val = feat_kv[1]
                if len(feat_kv) > 1 and int(feat_kv[0]) > dump_feature_cols:
                    continue
                dump_feats_slice[feat_idx] = feat_val
            for idx, feat in dump_feats_slice.items():
                cur_libSVM_format_feat.append(":".join([str(idx + 1), Utils.round_feat(feat)]))
        #非libSVM格式
        else:
            for idx, feat in enumerate(raw_dump_feats[:dump_feature_cols]):
                cur_libSVM_format_feat.append(":".join([str(idx + 1), Utils.round_feat(feat)]))
    # 空特征
    else:
        # 虽然col1是特征为空flag，但是线上取值浮点型常量0.0
        return None
    
    return "\t".join(cur_libSVM_format_feat)

def parse_sample_operator(x, country_code):
    def _parse_data(x):
        ret = []

        #拉字段
        line = x.strip().split("\t")
        if len(line) < SampleFiledCol["poi_info"]:
            return ret
        search_uid = line[SampleFiledCol["search_uid"]]
        query_info = json.loads(line[SampleFiledCol["query_info"]]) 
        poi_info = json.loads(line[SampleFiledCol["poi_info"]])

        #解析query侧特征
        query_data = {}
        sample_keys = []
        for feat_info in Config.QuerySampleInfo:
            default_val = feat_info.get("default_val", "")
            sample_key = feat_info["sample_key"]
            # 防止有重复的key
            if sample_key in sample_keys:
                continue
            if "parse_func" in feat_info:
                query_data = eval(feat_info["parse_func"](query_info, sample_key, default_val, query_data, country_code))
            else:
                query_data = parse_key_info(query_info, sample_key, default_val, query_data, country_code)
            sample_keys.append(sample_key)

        # 根据label过滤样本， label等于0是召回未展现样本， 大于0为展现点击样本
        label_filter_th = 0 if Config.ParseL1Sample else 1
        show_click_pois = [i for i in poi_info if 'p_label' in i and int(i['p_label']) >= label_filter_th]
        show_click_pois = sorted(show_click_pois, key=lambda x: int(x["p_show_pos"]))
        group_len = len(show_click_pois)
        group_prefix = "\t".join([search_uid, str(group_len)])

        #解析doc侧特征
        output_sample_data = []
        for poi_info in show_click_pois:
            sample_data = copy.deepcopy(query_data)
            for feat_info in Config.PoiSmapleInfo:
                default_val = feat_info.get("default_val", "")
                sample_key = feat_info["sample_key"]
                if "parse_func" in feat_info:
                    sample_data = eval(feat_info["parse_func"](poi_info, sample_key, default_val, sample_data, country_code))
                else:
                    sample_data = parse_key_info(poi_info, sample_key, default_val, sample_data, country_code)
            #处理lgbm特征（强过滤）
            cur_libSVM_format_feat = get_gbdt_feat(poi_info["p_ltrfeatures"], Config.DumpFeatureCols)
            if not cur_libSVM_format_feat:
                return ret
            
            # 落盘 parse_info \t searchid||uid \t group_len \t label \t show_pos \t idx1:feat1 \t idx2:feat2 \t ...
            output_sample_data.append(
                "\t".join([
                    json.dumps(sample_data),  #json.dumps(sample_data, ensure_ascii=False)
                    group_prefix,
                    poi_info["p_label"],
                    poi_info["p_show_pos"],
                    cur_libSVM_format_feat
                ]))

        if len(output_sample_data) > 0:
            ret.append([search_uid, "\n".join(output_sample_data)])
        
        return ret

    return _parse_data(x)




