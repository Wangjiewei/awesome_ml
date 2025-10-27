# encoding=utf8
import sys
import os
import json
import random
import datetime
from datetime import timedelta
from collections import defaultdict
from math import randians, cos, sin, asin, sqrt
from config import Config
reload(sys)
sys.setdefaultencoding('utf-8')

CountryLatLng = {
    "BR": {"max_lng": -34.7933, "min_lng": -73.9822, "max_lat": 5.2718, "min_lat": -33.7684},
    "MX": {"max_lng": -86.7105, "min_lng": -117.1278, "max_lat": 32.7187, "min_lat": 14.5326},
    "CO": {"max_lng": -34.7933, "min_lng": -73.9822, "max_lat": 5.2718, "min_lat": -33.7684},
    "CL": {"max_lng": -86.7105, "min_lng": -117.1278, "max_lat": 32.7187, "min_lat": 14.5326}
}

class Utils(object):
    @staticmethod
    def gen_key(searchid, uid):
        return "||".join([searchid, uid])
    
    @staticmethod
    def get_days_between(begin_date, end_date):
        date_list = []
        dt_begin = datetime.datetime.strptime(begin_date, "%Y%m%d")
        dt_end = datetime.datetime.strptime(end_date, "%Y%m%d")
        while dt_begin <= dt_end:
            date_str = dt_begin.strftime("%Y%m%d")
            date_list.append(date_str)
            dt_begin += timedelta(days=1)
        return date_list
    
    @staticmethod
    def get_feature_all_path(base_dir, country_code, begin_date, end_date):
        print(base_dir, country_code, begin_date, end_date)
        yyyymmdd_list = Utils.get_days_between(begin_date, end_date)
        print(yyyymmdd_list)
        input_hdfs_paths = ','.join([os.path.join(base_dir, d, country_code, '*') for d in yyyymmdd_list if d != '20230324'])
        return input_hdfs_paths
    
    @staticmethod
    def get_n_days_before(date_str, n, fmt="%Y%m%d"):
        dt = datetime.datetime.strptime(date_str, fmt)
        n_days_before = dt - timedelta(days=n)
        return n_days_before.strftime(fmt)
    
    @staticmethod
    def gen_loc(lat, lng):
        '''输入要求str
        '''
        return ','.join([str(lat), str(lng)])
    
    @staticmethod
    def round_feat(feat_str):
        '''减少落盘占用空间
        '''
        return str(round(float(feat_str), 7))
    
    @staticmethod
    def geo_hash_encode(lat, lng, precision):
        import geohash
        try:
            geo_code = geohash.encode(float(lat), float(lng), precision=precision)
        except:
            geo_code = "[UNK]"
        return geo_code
    
    @staticmethod















































































    @staticmethod
    def get_field_encode_idx(fm_sparse_dict, field_val, field_name):
        '''
        1) 在dict中，给field_val对应的idx
        2）不在dict中，给field_name UNK对应的idx
        '''
        encode_idx = fm_sparse_dict.get("||".join([str(field_val), field_name]), None)
        if not encode_idx:
            if field_name == "uid" and len(str(field_val)) >= 8:
                return int(Utils.calc_new_uid_index(str(field_val)))
            encode_idx = fm_sparse_dict.get("||".join(["UNK", field_name]))
        return int(encode_idx)
    
    @staticmethod
    def calc_new_uid_index(uid):
        #uid = str(int(uid[-8:]) % 948907)
        uid = str(int(uid[-8:]) % 14138257)
        return uid
    

    @staticmethod
    def get_token_id(tokenizer, query, name, addr, max_token_len=Config.TwinbertTokenLen):
        query_tokens = tokenizer.tokenize(query)
        name_tokens = tokenizer.tokenize(name)
        addr_tokens = tokenizer.tokenize(addr)
        # if len(query_tokens) > max_token_len - 2:


































    

















def LoadDataFromHDFS(sc, data_path, minPartitions=200):
    return sc.textFile(data_path, minPartitions=minPartitions)

def LoadDataFromHive(spark, start_date, end_date, country_code):
    sql = '''
        select 
            search_key,
            query_info,
            poi_info
        from sug_global.country_query_featureall
        where dt between {0} and {1} and country_code = "{2}"
    '''.format(start_date, end_date, country_code)
    print(sql)
    data_rdd = spark.sql(sql).rdd.map(lambda x: "\t".join( [ x["search_key"], x["query_info"], x["poi_info"]]))


def get_norm_str(raw_str):
    from normalizater import Normalizater
    nor = Normalizater()
    norm_str = nor.do_normalizate(raw_str)
    return norm_str

