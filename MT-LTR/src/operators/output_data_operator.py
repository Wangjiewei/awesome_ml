#coding:utf-8
import numpy as np
from pyspark.sql.types import *
from pyspark import SparkFiles
from config import Config
from id_convert_operator import *
from utils import LoadDataFromHDFS
import math
import json

MAX_CATE_LEN = 4

def parse_dump_feat(x):
    line = x.strip().split("\t")
    if len(line) < 4:
        return None
        #raise Exception("dump feat col is err!")
    return line


def get_gbm_num_leaves(gbm_model):
    '''辅助后续生成one hot leaf编码特征'''
    dump_info = gbm_model.dump_model()
    tree_info = dump_info["tree_info"]
    gbm_num_leaves = [tree_info[i]["num_leaves"] for i in range(Config.LeafTreeLimit)]
    one_hot_offset = [idx*num+1 for idx,num in enumerate(gbm_num_leaves)]
    return one_hot_offset

def get_id_feat(sample_info, sparse_dict, features):
    '''计算id类特征，主要通过featKey 算子'''
    import id_convert_operator
    sample_info = eval(sample_info)
    id_feats = []
    for feat_conf in features:
        sample_key = feat_conf["sample_key"]
        if "convert_id_func" in feat_conf:
            query_feat = eval("id_convert_operator."+feat_conf["convert_id_func"])(sparse_dict, sample_info, feat_conf)
        else:
            query_feat = Utils.get_field_encode_idx(sparse_dict, sample_info[sample_key], feat_conf['dict_key'])
        id_feats.append(query_feat)

    return id_feats

def padding_gbdt_feat(x):
    sample = x.strip("\n").split("\t")
    dump_feat_pads = ["0"] * (Config.DumpFeatureCols + 1)
    for feat in sample[5].split(";"):
        feat = feat.split(":")
        dump_feat_pads[int(feat[0])-1] = feat[1]
    sample[5] = ";".join([":".join((str(idx), val)) for idx, val in enumerate(dump_feat_pads)])
    return "\t".join(sample)

def get_struct_field(feat_keys, feat_type):
    feat_struct_fields = []
    for feat_info in feat_keys:
        tf_key = feat_info["sample_key"]
        tf_record_key = tf_key
        if feat_type in ("pos", "neg"):
            tf_record_key = "_".join((feat_type, tf_key))
        out_type = feat_info["out_type"]
        if out_type == "int":
            struct_field = StructField(tf_record_key, IntegerType(), False)
        elif out_type == "float":
            struct_field = StructField(tf_record_key, FloatType(), False)
        elif out_type == "array":
            struct_field = StructField(tf_record_key, ArrayType(IntegerType(), False))
        else:
            raise Exception("out_type err")
        feat_struct_fields.append(struct_field)
    if feat_type in ("pos", "neg"):
        feat_struct_fields.extend([
            StructField("_".join((feat_type, "pred_leaf")), ArrayType(IntegerType(), False)),
            StructField("_".join((feat_type, "onehot_pred_leaf")), ArrayType(IntegerType(), False)),
            StructField("_".join((feat_type, "tree_score")), FloatType(), False),
            StructField("_".join((feat_type, "gbdt_top_feat")), ArrayType(FloatType(), False)),
        ])
    return feat_struct_fields
def get_struct_by_outformat(outformat):
    schema =None
    if outformat == "query":
        struct_fields = [
            StructField("token_ids", ArrayType(IntegerType(), False)),
            StructField("segment_ids", ArrayType(IntegerType(), False)),
            StructField("mask_ids", ArrayType(IntegerType(), False)),
            StructField("poi_token_ids", ArrayType(IntegerType(), False)),
            StructField("city_id", IntegerType(), False),
            StructField("poiid", StringType(), False),
            StructField("name", StringType(), False),
            StructField("labels", IntegerType(), False),
        ]
        schema = StructType(struct_fields)
        
    elif outformat == "multy_feature":
        struct_fields = [
            StructField("token_ids", ArrayType(IntegerType(), False)),
            StructField("segment_ids", ArrayType(IntegerType(), False)),
            StructField("mask_ids", ArrayType(IntegerType(), False)),
            StructField("poi_token_ids", ArrayType(IntegerType(), False)),
            StructField("city_id", IntegerType(), False),
            StructField("poiid", StringType(), False),
            StructField("name", StringType(), False),
            StructField("query_type", IntegerType(), False),
            StructField("query_city_id", IntegerType(), False),
            StructField("query_geohash_id", IntegerType(), False),
            StructField("query_week", IntegerType(), False),
            StructField("query_day", IntegerType(), False),
            StructField("labels", IntegerType(), False),
        ]
        schema = StructType(struct_fields)
        
    elif outformat == "vqgan":
        struct_fields = [
            StructField("query_token_ids", ArrayType(IntegerType(), False)),
            StructField("query_segment_ids", ArrayType(IntegerType(), False)),
            StructField("query_mask_ids", ArrayType(IntegerType(), False)),
            StructField("poi_token_ids", ArrayType(IntegerType(), False)),
            StructField("poi_segment_ids", ArrayType(IntegerType(), False)),
            StructField("poi_mask_ids", ArrayType(IntegerType(), False)),
            StructField("labels", IntegerType(), False),
        ]
        schema = StructType(struct_fields)
        
    elif outformat == "twinbert":
        struct_fields = [
            StructField("query_input_ids", ArrayType(IntegerType(), False)),
            StructField("query_segment_ids", ArrayType(IntegerType(), False)),
            StructField("query_input_mask", ArrayType(IntegerType(), False)),
            StructField("pos_input_ids", ArrayType(IntegerType(), False)),
            StructField("pos_segment_ids", ArrayType(IntegerType(), False)),
            StructField("pos_input_mask", ArrayType(IntegerType(), False)),
            StructField("neg_input_ids", ArrayType(IntegerType(), False)),
            StructField("neg_segment_ids", ArrayType(IntegerType(), False)),
            StructField("neg_input_mask", ArrayType(IntegerType(), False)),
            #StructField("query_pos_input_ids", ArrayType(IntegerType(), False)),
            #StructField("query_pos_segment_ids", ArrayType(IntegerType(), False)),
            #StructField("query_pos_input_mask", ArrayType(IntegerType(), False)),
            #StructField("query_neg_input_ids", ArrayType(IntegerType(), False)),
            #StructField("query_neg_segment_ids", ArrayType(IntegerType(), False)),
            #StructField("query_neg_input_mask", ArrayType(IntegerType(), False)),
            StructField("label", IntegerType(), False),
        ]
        schema = StructType(struct_fields)
    elif outformat == "twinbert_l2":
        struct_fields = [
            StructField("pos_input_ids", ArrayType(IntegerType(), False)),
            StructField("pos_segment_ids", ArrayType(IntegerType(), False)),
            StructField("pos_input_mask", ArrayType(IntegerType(), False)),
            StructField("neg_input_ids", ArrayType(IntegerType(), False)),
            StructField("neg_segment_ids", ArrayType(IntegerType(), False)),
            StructField("neg_input_mask", ArrayType(IntegerType(), False)),
            StructField("label", IntegerType(), False),
        ]
            #StructField("query_input_ids", ArrayType(IntegerType(), False)),
            #StructField("query_segment_ids", ArrayType(IntegerType(), False)),
            #StructField("query_input_mask", ArrayType(IntegerType(), False)),
        schema = StructType(struct_fields)
        
    elif outformat == "twinbert_new":
        struct_fields = [
            StructField("query", StringType(), False),
            StructField("uid", IntegerType(), False),
            StructField("query_input_ids", ArrayType(IntegerType(), False)),
            StructField("query_segment_ids", ArrayType(IntegerType(), False)),
            StructField("query_input_mask", ArrayType(IntegerType(), False)),
            StructField("pos_poi_input_ids", ArrayType(IntegerType(), False)),
            StructField("pos_poi_segment_ids", ArrayType(IntegerType(), False)),
            StructField("pos_poi_input_mask", ArrayType(IntegerType(), False)),
            StructField("pos_input_ids", ArrayType(IntegerType(), False)),
            StructField("pos_segment_ids", ArrayType(IntegerType(), False)),
            StructField("pos_input_mask", ArrayType(IntegerType(), False)),
            StructField("pos_poiid", IntegerType(), False),
            StructField("pos_poiid_ori", StringType(), False),
            StructField("pos_match_rat", FloatType(), False),
            StructField("pos_match_len", IntegerType(), False),
            StructField("pos_name", StringType(), False),
            StructField("neg_poi_input_ids", ArrayType(IntegerType(), False)),
            StructField("neg_poi_segment_ids", ArrayType(IntegerType(), False)),
            StructField("neg_poi_input_mask", ArrayType(IntegerType(), False)),
            StructField("neg_input_ids", ArrayType(IntegerType(), False)),
            StructField("neg_segment_ids", ArrayType(IntegerType(), False)),
            StructField("neg_input_mask", ArrayType(IntegerType(), False)),
            StructField("neg_poiid", IntegerType(), False),
            StructField("neg_poiid_ori", StringType(), False),
            StructField("neg_match_rat", FloatType(), False),
            StructField("neg_match_len", IntegerType(), False),
            StructField("neg_name", StringType(), False),
            StructField("label", IntegerType(), False),
        ]
        schema = StructType(struct_fields)
        
    elif outformat == "twinbert_union":
        struct_fields = [
            StructField("group_key", StringType(), False),
            StructField("query", StringType(), False),
            StructField("q_type_feat", IntegerType(), False),
            StructField("time_feat", IntegerType(), False),
            StructField("query_token_ids", ArrayType(IntegerType(), False)),
            StructField("query_segment_ids", ArrayType(IntegerType(), False)),
            StructField("query_mask_ids", ArrayType(IntegerType(), False)),
            StructField("origin_labels", IntegerType(), False),
            StructField("labels", IntegerType(), False),
            StructField("poi_idx", IntegerType(), False),
            StructField("poi_token_ids", ArrayType(IntegerType(), False)),
            StructField("poi_segment_ids", ArrayType(IntegerType(), False)),
            StructField("poi_mask_ids", ArrayType(IntegerType(), False)),
            StructField("layer", IntegerType(), False),
            StructField("distance", IntegerType(), False),
            StructField("dense_feat", ArrayType(FloatType(), False)),
            StructField("nameaddr", StringType(), False),
        ]
        schema = StructType(struct_fields)

    return schema

def get_sparse_dict(dict_file):
    '''加载id映射词典'''
    sparse_dict = {}
    path = SparkFiles.get(dict_file)
    with open(path, 'r') as f:
        for l in f:
            idx, k = l.strip().split('\t')
            sparse_dict[k] = idx
    return sparse_dict

def calc_feat_qtiles(dmedian, featval):
        fftval = float(featval)
        if fftval == -1.0 or fftval <= 0 or dmedian <=0:
            return 0.0
        else:
            return round(math.log((1 + fftval) / (1 + dmedian)), 4)

def get_gbdt_top_feat(feat):
    '''计算gbdt top的特征，v-min/dmedian-min'''
    gbdt_feat = []
    for i in feat.split(";"):
        index, val = i.split(":")
        if index not in Config.GbdtFeatMap:
            continue
        feat_info = Config.GbdtFeatMap[index]
        median = feat_info[0]
        min_v = feat_info[1]
        new_feat = calc_feat_qtiles(median - min_v, float(val) - min_v)
        gbdt_feat.append(new_feat)
    return gbdt_feat

def get_gbdt_top_feat_new(feat):
    '''计算gbdt top的特征，v-min/dmedian-min'''
    gbdt_feat = []
    for index, val in enumerate(feat.split(";")):
        if index not in Config.DenseFeatMap:
            continue
        feat_info = Config.DenseFeatMap[index]
        median = feat_info[0]
        min_v = feat_info[1]
        new_feat = calc_feat_qtiles(median - min_v, float(val) - min_v)
        gbdt_feat.append(new_feat)
    return gbdt_feat

def f_to_s(data, sep=";"):
    data = [round(i, 6) for i in data]
    return sep.join(map(str, data))

def get_join_statistic(sample_info, q_type):
    default_data = [0] * 41
    dest_pid_city = f_to_s(sample_info.get("dest_pid_city", default_data))
    dest_pid_query_city = f_to_s(sample_info.get("dest_pid_query_city", default_data))
    dest_query_city = f_to_s(sample_info.get("dest_query_city", default_data))
    dest_query_geohash = f_to_s(sample_info.get("dest_query_geohash", default_data))
    start_query_geohash = f_to_s(sample_info.get("start_query_geohash", default_data))
    common_end_ac = sample_info.get("common_end_ac", 0)
    common_start_ac = sample_info.get("common_start_ac",0)
    person_end_ac = sample_info.get("person_end_ac", 0)
    person_start_ac = sample_info.get("person_start_ac", 0)
    common_total_ac = sample_info.get("common_total_ac",0)
    click_score = float(sample_info.get("click_score", None))
    is_click_score_empty = 1
    if not click_score:
        is_click_score_empty = 0
        click_score = 0.0
    if q_type == "0":
        if common_start_ac != 0:
            common_total_ac = 0
        common_end_ac = 0
        person_end_ac = 0
    else:
        if common_end_ac != 0:
            common_total_ac = 0
        common_start_ac = 0
        person_start_ac = 0
    return ";".join((";".join(map(str, (is_click_score_empty,click_score, 0, 0, 0, common_end_ac, 
                     common_start_ac, person_end_ac, person_start_ac, common_total_ac))), dest_query_city, 
                     dest_pid_query_city, dest_query_geohash, start_query_geohash, dest_pid_city))

def get_statistic_feat(sample_info, q_type, distance_f, match_rat):
    # dest_pid_city = f_to_s(sample_info.get("dest_pid_city", default_data))
    # dest_pid_city = dest_pid_query_city = dest_pid_query_geohash = dest_query_city = dest_query_geohash = start_query_geohash = common_start_ac = person_end_ac = person_start_ac = common_total_ac = ""
    default_data = [0] * 41
    dest_pid_city = f_to_s(sample_info.get("dest_pid_city", default_data))
    dest_pid_query_city = f_to_s(sample_info.get("dest_pid_query_city", default_data))
    dest_query_city = f_to_s(sample_info.get("dest_query_city", default_data))
    dest_query_geohash = f_to_s(sample_info.get("dest_query_geohash", default_data))
    start_query_geohash = f_to_s(sample_info.get("start_query_geohash", default_data))
    common_end_ac = sample_info.get("common_end_ac", 0)
    common_start_ac = sample_info.get("common_start_ac",0)
    person_end_ac = sample_info.get("person_end_ac", 0)
    person_start_ac = sample_info.get("person_start_ac", 0)
    common_total_ac = sample_info.get("common_total_ac",0)
    click_score = float(sample_info.get("click_score", 0.0))
    src_tag = sample_info.get("p_srctag", "")
    is_ts_recall = 0.0
    if "google_textsearch" in src_tag:
        is_ts_recall = 1.0
    is_click_score_empty = 0.0
    if click_score > 0.0:
        is_click_score_empty = 1.0
    if q_type == 0:
        dest_query_geohash = f_to_s(default_data)
        person_end_ac = 0.0
        common_end_ac = 0.0
    else:
        start_query_geohash = f_to_s(default_data)
        person_start_ac = 0.0
        common_start_ac = 0.0
    return ";".join((";".join(map(str, [distance_f, is_click_score_empty, click_score, is_ts_recall, common_end_ac, common_start_ac, person_end_ac, person_start_ac, common_total_ac])),
                    dest_query_city, dest_pid_query_city, dest_query_geohash, start_query_geohash, dest_pid_city, str(match_rat)))
    #return ";".join((";".join(map(str, [common_end_ac, common_start_ac, person_end_ac, person_start_ac, common_total_ac])),
    #                dest_query_city, dest_pid_query_city, dest_query_geohash, start_query_geohash, dest_pid_city, str(match_rat)))

def parse_ltr_new_feat(sample_info, match_rat):
    # 解析ltr_feat里需要的特征
    ltr_feat = sample_info["p_ltrfeatures"]
    new_feats = ["0.0" for i in range(215)]
    new_feats[-1] = match_rat
    for i in ltr_feat.strip(";").split(";"):
        index, v = i.split(":")
        index = int(index)
        v = float(v) if index != 28 else float(v) / 1000
        if index in Config.GbdtUseFeatIndex:
            new_feats[Config.GbdtUseFeatIndex[index]-1] = float(v)
    #return "\t".join([":".join(map(str, (index+1, v))) for index, v in enumerate(new_feats)])
    return ";".join([str(v) for index, v in enumerate(new_feats)])

def get_sim_feat(sample_data, text_nlp):
    query = sample_data["g_disp_query"]
    query_terms = query.split(" ")
    poi_name, poi_addr = sample_data["p_name_address"].split("|")
    name_terms = poi_name.split(" ")
    addr_terms = poi_addr.split(" ")
    cut_query_tfidf = text_nlp.str_to_bow(query_terms, "query_cut")
    ngram_query_tfidf = text_nlp.ngram_to_bow(query_terms, 2, "query_ngram")
    cut_name_tfidf = text_nlp.str_to_bow(name_terms, "name_cut")
    ngram_name_tfidf = text_nlp.ngram_to_bow(name_terms, 2, "name_ngram")
    cut_addr_tfidf = text_nlp.str_to_bow(addr_terms, "addr")
    sim_edit_name = text_nlp.sim_edit_distance(query, poi_name)
    sim_edit_addr = text_nlp.sim_edit_distance(query, poi_addr)
    sim_bm25_cut_name = text_nlp.simbm25(cut_query_tfidf, cut_name_tfidf, "name_cut")
    sim_bm25_cut_addr = text_nlp.simbm25(cut_query_tfidf, cut_addr_tfidf, "addr")
    sim_cos_cut_name, cut_name_query_sum_weight, cut_name_match_count_ratio, cut_match_weight_ratio, \
    cut_query_max_match_weight, cut_name_max_match_weight = text_nlp.sim_cosine(cut_query_tfidf, cut_name_tfidf)
    sim_cos_bi_name, bi_name_query_sum_weight, bi_name_match_count_ratio, bi_name_match_weight_ratio, \
    bi_query_max_match_weight, bi_name_max_match_weight = text_nlp.sim_cosine(ngram_query_tfidf, ngram_name_tfidf)
    query_split_len, name_split_len, addr_split_len, query_len,name_len, addr_len = [len(i) for i in [query_terms, name_terms, addr_terms, query, poi_name, poi_addr]]
    
    sim_info = f_to_s([1.0, sim_cos_cut_name, 
                            cut_name_query_sum_weight, 
                            cut_name_match_count_ratio, 
                            cut_match_weight_ratio,
                            cut_query_max_match_weight,
                            cut_name_max_match_weight,
                            sim_cos_bi_name,
                            bi_name_query_sum_weight,
                            bi_name_match_count_ratio,
                            bi_name_match_weight_ratio,
                            bi_query_max_match_weight,
                            bi_name_max_match_weight,
                            sim_bm25_cut_name,
                            sim_bm25_cut_addr,
                            sim_edit_name,
                            sim_edit_addr,
                            query_split_len,
                            name_split_len,
                            addr_split_len,
                            query_len,
                            name_len,
                            addr_len, ])
    return sim_info

def format_gbdt(dump_feature_rdd, out_partitions, save_path, cal_sim_feat=False):
    '''格式化输出 gbdt样本'''
    def _parse_sample(x):
        data = parse_dump_feat(x)
        if not data:
            return []
        sample_info, traceid_uid, group_len, label, show_pos, gbdt_feat = data
        return [[traceid_uid, [[sample_info, group_len, label, show_pos, gbdt_feat]]]]

    def _reduce_merge(x):
        sample_key = x[0]
        gbdt_feats = sorted(x[1], key=lambda k: k[3])
        #gbdt_feats = set(gbdt_feats)
        ret = []
        poiids = []
        group_len = str(len(gbdt_feats))
        tmp_ret = []
        for feat in gbdt_feats:
            sp_info = json.loads(feat[0])
            if sp_info["p_poi_id"] in poiids:
                continue
            poiids.append(sp_info["p_poi_id"])
            tmp_ret.append([feat[0], sample_key, group_len, feat[2], feat[4].replace(";", "\t")])
        group_len = len(tmp_ret)
        ret = ["\t".join(i[:2] + [str(group_len)] + i[3:]) for i in tmp_ret]
        return ["\n".join(ret)]
    def _calc_sim_feat(x):
        from feature_nlp import TextNlp
        query_cut_idf_f = SparkFiles.get("eg_query_cut.dict")
        query_ngram_idf_f = SparkFiles.get("eg_query_ngram.dict")
        query_dict_conf = SparkFiles.get("dict.config")
        poi_name_cut_idf_f = SparkFiles.get("eg_name_cut.dict")
        poi_addr_cut_idf_f = SparkFiles.get("eg_addr_cut.dict")
        poi_name_ngram_idf_f = SparkFiles.get("eg_name_ngram.dict")
        poi_dict_conf = SparkFiles.get("poi_dict.config")
        text_nlp = TextNlp(query_cut_idf_f, query_ngram_idf_f, query_dict_conf,poi_name_cut_idf_f,poi_name_ngram_idf_f,poi_addr_cut_idf_f, poi_dict_conf)
        data = x.strip().split("\t")
        sample_info = json.loads(data[0])
        searchid = data[1]
        group_len = data[2]
        label = str(data[3])
        position = str(data[4])
        q_type = 1.0 if sample_info["g_query_type"] == 1 else 0
        distance_feat, distance_f = Utils.geodistance(sample_info["g_plng"], sample_info["g_plat"], sample_info["p_poi_lng"], sample_info["p_poi_lat"])
        sim_info = get_sim_feat(sample_info,)
        other_info = ";".join([0,q_type, 0, distance_f, 0, 0]) # rank_pos, q_type, distance, empty_val
        stat_feat = get_join_statistic(sample_info, sample_info["g_query_type"])
        all_feat = ";".join((sim_info, other_info, stat_feat))
        dense_feat = ";".join([":".join((str(idx), i)) for idx, i in enumerate(all_feat.split(";"))])
        return [[searchid, [[data[0], group_len, label, position, dense_feat]]]]
    if cal_sim_feat:
        dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _calc_sim_feat(x)) \
            .reduceByKey(lambda v1, v2: v1 + v2) \
            .flatMap(lambda x: _reduce_merge(x))
    else:
        dump_feature_rdd = dump_feature_rdd.flatMap(lambda x: _parse_sample(x)) \
            .reduceByKey(lambda v1, v2: v1 + v2) \
            .flatMap(lambda x: _reduce_merge(x))
    dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path)


def format_deepfm(sc, spark, dump_feature_rdd, out_partitions, save_path, dict_file, model_file):
    '''格式化输出deepfm格式样本'''

    def _parse_sample(one_part_data):
        import numpy as np
        import lightgbm as lgbm

        def _proccess_batch_data(batch_data, one_hot_offset, gbm_model):
            '''batch 处理，提升gbdt预测性能'''
            gbdt_feat = np.array([[dd.split(':')[1] for dd in d.split('\t')[5].split(";")] for d in batch_data], dtype=np.float32)
            #gbdt_feat = np.array([["0.0"]*525 for d in batch_data], dtype=np.float32)
            # 算leaf节点编码特征
            one_hot_offset_arr = np.array(one_hot_offset)
            pred_leaf = gbm_model.predict(gbdt_feat, pred_leaf=True, num_iteration=Config.LeafTreeLimit)
            one_hot_pred_leaf = pred_leaf + one_hot_offset_arr
            #one_hot_pred_leaf = pred_leaf
            # 算tree score特征
            tree_scores = gbm_model.predict(gbdt_feat)
            ret = []
            for i, data in enumerate(batch_data):
                sample_info, search_uid, group_len, label, show_pos, gbdt_feats = parse_dump_feat(data)
                pred_leaf_dump = [int(pred_leaf[i][idx]) for idx in range(0, pred_leaf.shape[1])]
                one_hot_pred_leaf_dump = [int(one_hot_pred_leaf[i][idx]) for idx in range(0, one_hot_pred_leaf.shape[1])]
                tree_scores_dump = float(tree_scores[i])
                id_feat_encode = get_id_feat(sample_info, sparse_dict, Config.QueryFeatKey + Config.PoiFeatKey)
                gbdt_top_feat = get_gbdt_top_feat(gbdt_feats)
                ret.append([search_uid, [[int(label)-1,] + id_feat_encode + [pred_leaf_dump, one_hot_pred_leaf_dump, tree_scores_dump, gbdt_top_feat]]])

            return ret
        sparse_dict = get_sparse_dict(dict_file)
        # 加载模型
        gbm_model = lgbm.Booster(model_file=model_file)
        one_hot_offset = get_gbm_num_leaves(gbm_model)
        BATCH_SIZE = 100000
        batch_data = []
        ret = []
        for idx, data in enumerate(one_part_data):
            if idx > 0 and idx % BATCH_SIZE == 0:
                batch_res = _proccess_batch_data(batch_data, one_hot_offset, gbm_model)
                ret.extend(batch_res)
                batch_data = []
            batch_data.append(data)
        if len(batch_data) > 0:
            batch_res = _proccess_batch_data(batch_data, one_hot_offset, gbm_model)
            ret.extend(batch_res)

        return ret

    def _reduce_merge(x, query_keys_len):
        ret = []
        search_uid = x[0]
        feats = sorted(x[1], key=lambda i: i[0], reverse=True)
        pos_feat = feats[0]
        for feat in feats:
            feat_res = [search_uid, feat[0]]
            feat_res.extend(pos_feat[1:])
            neg_feat = feat[query_keys_len+1:]
            feat_res.extend(neg_feat)
            ret.append(feat_res)
        return ret

    def _get_out_schema():
        struct_fields = [
            StructField("traceid_uid", StringType(), False),
            StructField("label", IntegerType(), False),
        ]
        query_field = get_struct_field(Config.QueryFeatKey, "")
        pos_poi_field = get_struct_field(Config.PoiFeatKey, "pos")
        neg_poi_field = get_struct_field(Config.PoiFeatKey, "neg")
        struct_fields.extend(query_field)
        struct_fields.extend(pos_poi_field)
        struct_fields.extend(neg_poi_field)
        return StructType(struct_fields)

    query_keys_len = len(Config.QueryFeatKey)
    #dump_feature_rdd = dump_feature_rdd.map(lambda x: padding_gbdt_feat(x)) \
    dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x)) \
        .reduceByKey(lambda v1, v2: v1+v2) \
        .flatMap(lambda x: _reduce_merge(x, query_keys_len))
    schema = _get_out_schema()
    #.mapPartitions(lambda x: _parse_sample(x)) \
    #dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x)) \
    print(schema)
    #dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path+"_raw")
    spark\
        .createDataFrame(dump_feature_rdd, schema) \
        .repartition(out_partitions) \
        .write.mode("overwrite").format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .save(save_path)
    # show result case
    spark\
        .read.format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .load(save_path + '/*') \
        .show()

def format_bert(sc, spark, dump_feature_rdd, out_partitions, save_path, dict_file, model_file):
    '''格式化输出deepfm格式样本'''

    def _parse_sample(one_part_data):
        import numpy as np
        import lightgbm as lgbm
        import tokenization
        tokenizer = tokenization.FullTokenizer(vocab_file="./vocab.txt", do_lower_case=True)

        def _proccess_batch_data(batch_data, one_hot_offset, gbm_model):
            '''batch 处理，提升gbdt预测性能'''
            gbdt_feat = np.array([[dd.split(':')[1] for dd in d.split('\t')[5].split(";")] for d in batch_data], dtype=np.float32)
            #gbdt_feat = np.array([["0.0"]*525 for d in batch_data], dtype=np.float32)
            # 算leaf节点编码特征
            one_hot_offset_arr = np.array(one_hot_offset)
            pred_leaf = gbm_model.predict(gbdt_feat, pred_leaf=True, num_iteration=Config.LeafTreeLimit)
            one_hot_pred_leaf = pred_leaf + one_hot_offset_arr
            # 算tree score特征
            tree_scores = gbm_model.predict(gbdt_feat)
            ret = []
            for i, data in enumerate(batch_data):
                sample_info, search_uid, group_len, label, show_pos, gbdt_feats = parse_dump_feat(data)
                pred_leaf_dump = [int(pred_leaf[i][idx]) for idx in range(0, pred_leaf.shape[1])]
                one_hot_pred_leaf_dump = [int(one_hot_pred_leaf[i][idx]) for idx in range(0, one_hot_pred_leaf.shape[1])]
                tree_scores_dump = float(tree_scores[i])
                id_feat_encode = get_id_feat(sample_info, sparse_dict, Config.BertQueryFeatKey+Config.BertPoiFeatKey)
                gbdt_top_feat = get_gbdt_top_feat(gbdt_feats)
                token_ids, segment_ids, mask_ids = covert_bert_token(sparse_dict, sample_info, tokenizer, one_hot_pred_leaf_dump)
                ret.append([search_uid, [[int(label)-1,] + id_feat_encode + [pred_leaf_dump, one_hot_pred_leaf_dump, tree_scores_dump, gbdt_top_feat, token_ids, segment_ids, mask_ids]]])

            return ret
        sparse_dict = get_sparse_dict(dict_file)
        # 加载模型
        gbm_model = lgbm.Booster(model_file=model_file)
        one_hot_offset = get_gbm_num_leaves(gbm_model)
        BATCH_SIZE = 100000
        batch_data = []
        ret = []
        for idx, data in enumerate(one_part_data):
            if idx > 0 and idx % BATCH_SIZE == 0:
                batch_res = _proccess_batch_data(batch_data, one_hot_offset, gbm_model)
                ret.extend(batch_res)
                batch_data = []
            batch_data.append(data)
        if len(batch_data) > 0:
            batch_res = _proccess_batch_data(batch_data, one_hot_offset, gbm_model)
            ret.extend(batch_res)

        return ret

    def _reduce_merge(x, query_keys_len):
        ret = []
        search_uid = x[0]
        feats = sorted(x[1], key=lambda i: i[0], reverse=True)
        pos_feat = feats[0]
        for feat in feats:
            feat_res = [search_uid, feat[0]]
            feat_res.extend(pos_feat[1:])
            neg_feat = feat[query_keys_len+1:]
            feat_res.extend(neg_feat)
            ret.append(feat_res)
        return ret

    def _get_out_schema():
        struct_fields = [
            StructField("traceid_uid", StringType(), False),
            StructField("label", IntegerType(), False),
        ]
        query_field = get_struct_field(Config.BertQueryFeatKey, "")
        pos_poi_field = get_struct_field(Config.BertPoiFeatKey, "pos")
        neg_poi_field = get_struct_field(Config.BertPoiFeatKey, "neg")
        struct_fields.extend(query_field)
        struct_fields.extend(pos_poi_field)
        struct_fields.extend([
            StructField("pos_token_id", ArrayType(IntegerType(), False)),
            StructField("pos_segment_id", ArrayType(IntegerType(), False)),
            StructField("pos_mask_id", ArrayType(IntegerType(), False)),
        ])
        struct_fields.extend(neg_poi_field)
        struct_fields.extend([
            StructField("neg_token_id", ArrayType(IntegerType(), False)),
            StructField("neg_segment_id", ArrayType(IntegerType(), False)),
            StructField("neg_mask_id", ArrayType(IntegerType(), False)),
        ])
        return StructType(struct_fields)

    query_keys_len = len(Config.BertQueryFeatKey)
    #dump_feature_rdd = dump_feature_rdd.map(lambda x: padding_gbdt_feat(x)) \
    dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x)) \
        .reduceByKey(lambda v1, v2: v1+v2) \
        .flatMap(lambda x: _reduce_merge(x, query_keys_len))
    schema = _get_out_schema()
    print(schema)
    #dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path+"_raw")
    spark\
        .createDataFrame(dump_feature_rdd, schema) \
        .repartition(out_partitions) \
        .write.mode("overwrite").format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .save(save_path)
    # show result case
    spark\
        .read.format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .load(save_path + '/*') \
        .show()

def format_twinbert(sc, spark, dump_feature_rdd, out_partitions, save_path, outformat, poiid_idx_path, vocab_file, dict_file, model_file=None):
    def out_format(x):
        poiid = x[0]
        query_feat = x[1][0]
        poiid_idx = x[1][1]
        if not poiid_idx:
            query_feat.append(0)
            #return []
            return query_feat
        query_feat.append(int(poiid_idx))
        #query_feat.append(int(poiid))
        return query_feat

    def _parse_sample(data, parse_type="poi"):

        import numpy as np
        import lightgbm as lgbm
        import tokenization
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        text_nlp = None
        gbm_model = None#lgbm.Booster(model_file=model_file)
        one_hot_offset = None#get_gbm_num_leaves(gbm_model)
        if parse_type == "twinbert_union":
            # 离线文本相关性特征
            from feature_nlp import TextNlp
            query_cut_idf_f = SparkFiles.get("cut.dict")
            query_ngram_idf_f = SparkFiles.get("ngram.dict")
            query_dict_conf = SparkFiles.get("dict.config")
            poi_name_cut_idf_f = SparkFiles.get("displayname.dict")
            poi_addr_cut_idf_f = SparkFiles.get("address.dict")
            poi_name_ngram_idf_f = SparkFiles.get("displayname_ngram.dict")
            poi_dict_conf = SparkFiles.get("poi_dict.config")
            text_nlp = TextNlp(query_cut_idf_f, query_ngram_idf_f, query_dict_conf,poi_name_cut_idf_f,poi_name_ngram_idf_f,poi_addr_cut_idf_f, poi_dict_conf)
            gbm_model = lgbm.Booster(model_file=model_file)
            one_hot_offset = get_gbm_num_leaves(gbm_model)
            
        def get_poi_data(data):
            # rl 训练生成 poi 数据
            res = []
            ac_flag = True  # 是否拼接AC信息
            for line in data:
                fields = line.strip().split('||')
                poiid, name, addr, lng, lat, area, country, area_id, poi_geocode_id, click_score, category_ids, country_str, ac = fields
                ac_len = 32 - len(name.split(" ")) - len(addr.split(" "))
                ac_info = " ".join(ac.split(" ")[:ac_len])
                if(ac_flag):
                    token_ids, segment_ids, mask_ids = convert_poi_token_ids("|".join((name, " ".join((addr, ac_info)))), Config.TwinbertTokenLen, tokenizer)
                else:
                    token_ids, segment_ids, mask_ids = convert_poi_token_ids("|".join((name, addr)), Config.TwinbertTokenLen, tokenizer)

                poi_category_ids = category_ids.split("#")
                len_poi_category_ids = len(poi_category_ids)
                if len_poi_category_ids > MAX_CATE_LEN:
                    poi_category_ids = poi_category_ids[0:MAX_CATE_LEN]
                    poi_category_mask = [1] * MAX_CATE_LEN
                else:
                    poi_category_ids = poi_category_ids + [0] * (MAX_CATE_LEN - len_poi_category_ids)
                    poi_category_mask = [1] * len_poi_category_ids + [0] * (MAX_CATE_LEN - len_poi_category_ids)

                #res.append("\t".join((poiid, area, ";".join(map(str, token_ids)),int(area_id),int(poi_geocode_id),float(click_score),list(map(int,poi_category_ids)), poi_category_mask)))
                res.append("\t".join((str(poiid), str(area), ";".join(map(str, token_ids)),str(area_id),str(poi_geocode_id),str(click_score),",".join(map(str,poi_category_ids)),",".join(map(str, poi_category_mask)) )))
            return res
        def get_query_data(data):
            # rl 训练生成 query 数据
            res = []
            for line in data:
                fields = line.strip('\r\n').split('\t')
                g_searchid, g_country_code, cityid, poiid, query, pstnm, neg_list_str, g_birth_time, uniq_id, correct_type, poi_queue = fields[1].strip().split('##')[:11]
                #if('_test' in save_path and cityid != '55000199'): continue  # only for test set
                if('_test' in save_path and cityid != '52090100'): continue  # only for test set
                if(len(query) < 15): continue
                #query, pstnm, cityid, poiid = fields[:4]
                token_ids, segment_ids, mask_ids = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer)
                poi_token_ids, poi_segment_ids, poi_mask_ids = convert_poi_token_ids(pstnm, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                res.append([poiid, [token_ids, segment_ids, mask_ids,poi_token_ids, int(cityid), poiid, pstnm]])
            return res
        def get_multy_feature_train(data):
            # twinbert 多特征模型
            res = []
            for line in data:
                fields = line.strip('\r\n').split('\t')
                g_searchid, g_country_code, cityid, poiid, query, pstnm, neg_list_str, g_birth_time, uniq_id, correct_type, poi_queue, g_query_type, g_area_id,g_geohash_id, g_week_bucket, g_day_bucket = fields[1].strip().split('##')[:16]
                if('_test' in save_path and cityid != '55000199'): continue  # only for test set
                #if('_test' in save_path and cityid != '52090100'): continue  # only for test set
                if('_test' in save_path and correct_type != 'origin'): continue  # delect deep_correct sample
                #if(len(query) < 15): continue
                #query, pstnm, cityid, poiid = fields[:4]
                token_ids, segment_ids, mask_ids = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer)
                poi_token_ids, poi_segment_ids, poi_mask_ids = convert_poi_token_ids(pstnm, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                res.append([poiid, [token_ids, segment_ids, mask_ids,poi_token_ids, int(cityid), poiid, pstnm, int(g_query_type), int(g_area_id), int(g_geohash_id), int(g_week_bucket), int(g_day_bucket)]])

            return res

        def get_vqgan_train_hardneg(data):
            # 生成 vqgan 训练数据
            res = []
            for line in data:
                fields = line.strip('\r\n').split("\t")
                if len(fields) < 6:
                    continue
                query_idx, query, poiid, poi_idx, name_addr, label_str = fields
                query_idx = int(query_idx)
                poi_idx = int(poi_idx)
                #if label_str == "negative":
                #    continue
                token_ids, segment_ids, mask_ids, query_tokens = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                poi_token_ids, poi_segment_ids, poi_mask_ids, poi_tokens = convert_poi_token_ids(name_addr, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                label = 0
                if label_str == "positive" or label_str == "hard_negative": # 这里hard negative作为正例
                    label = 1
                origin_labels = 0
                if label_str == "positive":
                    origin_labels = 1
                #res.append([query_idx, poi_idx, token_ids, segment_ids, mask_ids, poi_token_ids, poi_segment_ids, poi_mask_ids, label, origin_labels, query, name_addr, query_tokens, poi_tokens])
                res.append([query_idx, poi_idx, token_ids, segment_ids, mask_ids, poi_token_ids, poi_segment_ids, poi_mask_ids, label, origin_labels])
            return res
        def get_twinbert_train(data):
            # 生成 twinbert 训练数据
            res = []
            for line in data:
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    continue
                query, pos_name, neg_name = fields[:3]
                query_token_ids, query_segment_ids, query_mask_ids = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer)
                pos_token_ids, pos_segment_ids, pos_mask_ids = convert_poi_token_ids(pos_name, Config.TwinbertTokenLen, tokenizer, is_pad=True) 
                neg_token_ids, neg_segment_ids, neg_mask_ids = convert_poi_token_ids(neg_name, Config.TwinbertTokenLen, tokenizer, is_pad=True) 
                #query_pos_token_ids, query_pos_segment_ids, query_pos_mask_ids = convert_poi_token_ids("|".join((query, pos_name.replace("|"," "))), Config.TwinbertTokenLen, tokenizer, is_pad=True)
                #query_neg_token_ids, query_neg_segment_ids, query_neg_mask_ids = convert_poi_token_ids("|".join((query, neg_name.replace("|"," "))), Config.TwinbertTokenLen, tokenizer, is_pad=True)
                #pad_index = query_token_ids.index(5) + 1
                #query_pos_token_ids = query_token_ids[:pad_index] + pos_token_ids[1:] + [0] + query_token_ids[pad_index:]
                #query_pos_segment_ids = query_segment_ids[:pad_index] + pos_segment_ids[1:] + [0] + query_segment_ids[pad_index:]
                #query_pos_mask_ids = query_mask_ids[:pad_index] + pos_mask_ids[1:] + [0] + query_mask_ids[pad_index:]
                #query_neg_token_ids = query_token_ids[:pad_index] + neg_token_ids[1:] + [0] + query_token_ids[pad_index:]
                #query_neg_segment_ids = query_segment_ids[:pad_index] + neg_segment_ids[1:] + [0] + query_segment_ids[pad_index:]
                #query_neg_mask_ids = query_mask_ids[:pad_index] + neg_mask_ids[1:] + [0] + query_mask_ids[pad_index:]
                #res.append([query_token_ids, query_segment_ids, query_mask_ids, pos_token_ids, pos_segment_ids, pos_mask_ids, neg_token_ids, neg_segment_ids, neg_mask_ids,query_pos_token_ids, query_pos_segment_ids, query_pos_mask_ids, query_neg_token_ids, query_neg_segment_ids,query_neg_mask_ids, 1])
                res.append([query_token_ids, query_segment_ids, query_mask_ids, pos_token_ids, pos_segment_ids, pos_mask_ids, neg_token_ids, neg_segment_ids, neg_mask_ids, 1])
            return res
        
        def get_twinbert_train_l2(data):
            # 直接将点展样本生成 twinbert 训练样本
            res = []
            for sample in data:
                samples = sample.split("\n") 
                pos_name = ""
                for line in samples:
                    line = parse_dump_feat(line)
                    if line == None:
                        continue
                    sample_info, search_uid, group_len, label, show_pos, gbdt_feats = line
                    sample_info = eval(sample_info)
                    if label == "2":
                        pos_name = sample_info["p_name_address"]
                if pos_name == "":
                    continue
                for line in samples:
                    line = parse_dump_feat(line)
                    if line == None:
                        continue
                    sample_info, search_uid, group_len, label, show_pos, gbdt_feats = line
                    sample_info = eval(sample_info)
                    if label == "2":
                        continue
                    query = sample_info["g_disp_query"]
                    neg_name = sample_info["p_name_address"]
                    #query_token_ids, query_segment_ids, query_mask_ids = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer)
                    pos_token_ids, pos_segment_ids, pos_mask_ids = convert_poi_token_ids("|".join((query,pos_name)), Config.TwinbertTokenLen*2, tokenizer, is_pad=True) 
                    neg_token_ids, neg_segment_ids, neg_mask_ids = convert_poi_token_ids("|".join((query, neg_name)), Config.TwinbertTokenLen*2, tokenizer, is_pad=True) 
                    #res.append([query_token_ids, query_segment_ids, query_mask_ids, pos_token_ids, pos_segment_ids, pos_mask_ids,neg_token_ids,neg_segment_ids, neg_mask_ids, 1])
                    res.append([pos_token_ids, pos_segment_ids, pos_mask_ids,neg_token_ids,neg_segment_ids, neg_mask_ids, 1])
            return res

        def merge_token_feat(token_ids, segment_ids, mask_ids, feat_token_ids, feat_segment_ids, feat_mask_ids, max_token_len):
            token_ids.extend(feat_token_ids)
            segment_ids.extend(feat_segment_ids)
            mask_ids.extend(feat_mask_ids)
            new_token_len = len(token_ids)
            if new_token_len < max_token_len:
                padding_ids = [0] * (max_token_len - new_token_len)
                token_ids.extend(padding_ids)
                segment_ids.extend(padding_ids)
                mask_ids.extend(padding_ids)
            return token_ids, segment_ids, mask_ids

        def get_twinbert_train_rank(data):
            sparse_dict = get_sparse_dict(dict_file)
            res = []
            token_len = Config.TwinbertTokenLen + 14
            for line in data:
                fields = line.strip().split("\t")
                if len(fields) < 8:
                    continue
                #city, traceid, query, uid, q_latlng, time_str, qtype, q_geo, poi_infos = fields
                city, query, uid, q_latlng, time_str, qtype, q_geo, poi_infos = fields
                if city != "52090100":
                    continue
                q_lat, q_lng = q_latlng.split("|")
                uid_feat = Utils.get_field_encode_idx(sparse_dict, uid, "uid")
                q_geo_feat = Utils.get_field_encode_idx(sparse_dict, q_geo, "geo")
                q_type_feat = 0 if qtype == "0" else 1
                q_type_feat = Utils.get_field_encode_idx(sparse_dict, str(q_type_feat), "qtype")
                time_hour = int(time_str.split(" ")[1].split(":")[0])
                if time_hour >5 and time_hour < 12:
                    time_feat = 0
                elif time_hour >= 12 and time_hour < 18:
                    time_feat = 1
                else:
                    time_feat = 2
                time_feat = Utils.get_field_encode_idx(sparse_dict, str(time_feat), "hour")
                # 生成query侧 tokenids
                query_token_ids, query_segment_ids, query_mask_ids = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                query_feat_tokens = [q_geo_feat, 5, q_type_feat, 5, time_feat,5]
                query_feat_segments = [2,2, 3,3, 4,4]
                query_feat_masks = [1]*6
                query_token_ids, query_segment_ids, query_mask_ids = merge_token_feat(query_token_ids, query_segment_ids, query_mask_ids, query_feat_tokens, query_feat_segments, query_feat_masks, Config.TwinbertTokenLen + 12)
                pos_feats = []
                neg_feats = []
                count = 0
                hard_feats = []
                query_set = set(query.split(" "))
                query_word_len = float(len(query_set))
                for info in poi_infos.split("||"):
                    info = info.split("##")
                    if len(info) < 7:
                        continue
                    label, show_pos, poiid, nameaddr, lat, lng, cate,layer, p_geo = info
                    poiid_feat = Utils.get_field_encode_idx(sparse_dict, poiid, "poiid")
                    poi_geo_feat = Utils.get_field_encode_idx(sparse_dict, p_geo, "geo")
                    poi_cate_feat = Utils.get_field_encode_idx(sparse_dict, cate, "cate")
                    poi_layer_feat = Utils.get_field_encode_idx(sparse_dict, layer, "layer")
                    distance_feat = Utils.geodistance(q_lng, q_lat, lng, lat)
                    name_addr_set = set(nameaddr.replace("|", " ").split(" "))
                    match_rat = len(query_set&name_addr_set) / query_word_len
                    match_len = len(query_set&name_addr_set)
                    if match_len > 20:
                        match_len = 20
                    poi_token_ids, poi_segment_ids, poi_mask_ids = convert_poi_token_ids(nameaddr, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                    poi_feat_tokens = [0, 0, 0, 0, 0, 0, poi_geo_feat, 5, poi_cate_feat, 5, poi_layer_feat, 5]
                    poi_feat_segments = [0,0,0,0,0,0, 5,5, 6,6, 7,7]
                    poi_feat_masks = [0] * 6 + [1]*6
                    poi_token_ids, poi_segment_ids, poi_mask_ids = merge_token_feat(poi_token_ids, poi_segment_ids, poi_mask_ids, poi_feat_tokens, poi_feat_segments, poi_feat_masks, Config.TwinbertTokenLen + 12)
                    # 生成交互模型的特征
                    nameaddr = nameaddr.split("|")
                    token_ids, segment_ids, mask_ids = Utils.get_token_id(tokenizer, query, nameaddr[0], nameaddr[1])
                    feat_tokends = [q_geo_feat, 5, q_type_feat, 5, time_feat, 5, poi_geo_feat, 5, poi_cate_feat, 5, poi_layer_feat, 5, distance_feat, 5]
                    feat_segments = [2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8]
                    feat_masks = [1]*14
                    token_ids, segment_ids, mask_ids = merge_token_feat(token_ids, segment_ids,mask_ids, feat_tokends, feat_segments, feat_masks, Config.TwinbertTokenLen + 14)
                    new_label = 0
                    if label == "2" or (show_pos != -1 and count < 3):
                        new_label = 1
                    count += 1

                    poi_feats = [poi_token_ids, poi_segment_ids, poi_mask_ids, token_ids, segment_ids, mask_ids, poiid_feat, poiid, match_rat, match_len, nameaddr, int(new_label)]
                    if label == "2":
                        pos_feats = poi_feats
                    else:
                        if show_pos != "-1":
                            hard_feats.append(poi_feats)
                        else:
                            neg_feats.append(poi_feats)
                if not pos_feats:
                    continue
                #if len(hard_feats) < 4:
                #    hard_feats.extend(neg_feats[:4-len(hard_feats)])
                #else:
                #    hard_feats = hard_feats[:4]
                hard_feats.append(pos_feats)
                hard_feats.extend(neg_feats)
                for neg_feat in hard_feats:
                    #res.append(["|".join((traceid,query)), uid_feat, query_token_ids, query_segment_ids, query_mask_ids] + pos_feats[:-1] + neg_feat)
                    res.append([query, uid_feat, query_token_ids, query_segment_ids, query_mask_ids] + pos_feats[:-1] + neg_feat)
                #break

            return res

        
        
        def _process_batch(batch_data):
            gbdt_feat = np.array([d[1][0][-1] for d in batch_data], dtype=np.float32)
            #gbdt_feat = np.array([["0.0"]*525 for d in batch_data], dtype=np.float32)
            # 算leaf节点编码特征
            one_hot_offset_arr = np.array(one_hot_offset)
            pred_leaf = gbm_model.predict(gbdt_feat, pred_leaf=True, num_iteration=Config.LeafTreeLimit)
            one_hot_pred_leaf = pred_leaf + one_hot_offset_arr + 66453
            gbdt_seg_ids = [9]*300
            gbdt_mask_ids = [1] * 300
            new_batch_data = []
            for idx, d in enumerate(batch_data):
                new_data = d[:-9]
                token_ids = d[1][0][-9]
                segment_ids = d[1][0][-8]
                mask_ids = d[1][0][-7]
                token_ids.extend(one_hot_pred_leaf.tolist()[idx])
                segment_ids.extend(gbdt_seg_ids)
                mask_ids.extend(gbdt_mask_ids)
                new_data.extend([token_ids, segment_ids, mask_ids])
                new_data.extend(d[-6:])
                new_batch_data.append(new_data)
            return new_batch_data
        

        
        def get_id_feat(sample_data, sparse_dict):
            q_lat = sample_data["g_plat"]
            q_lng = sample_data["g_plng"]
            time_str = sample_data["g_timestamp"]
            #poiid_feat = Utils.get_field_encode_idx(sparse_dict, sample_data["p_poi_id"], "poiid")
            #p_geo_feat = Utils.get_field_encode_idx(sparse_dict, sample_data["p_geohash"], "geo")
            #p_cate_feat = Utils.get_field_encode_idx(sparse_dict, sample_data["p_category"], "cate")
            #p_layer_feat = Utils.get_field_encode_idx(sparse_dict, sample_data["layer"], "layer")
            p_layer_feat = int(sample_data["p_layer"])
            # if city != "52090100":
            #     continue
            # q_lat, q_lng = q_latlng.split("|")
            #uid_feat = Utils.get_field_encode_idx(sparse_dict, sample_data["g_uid"], "uid")
            #q_geo_feat = Utils.get_field_encode_idx(sparse_dict, sample_data["q_geohash"], "geo")
            q_type_feat = 0 if sample_data["g_query_type"] == "0" else 1
            time_hour = int(time_str.split(" ")[1].split(":")[0])
            if time_hour >5 and time_hour < 12:
                time_feat = 0
            elif time_hour >= 12 and time_hour < 18:
                time_feat = 1
            else:
                time_feat = 2
            #time_feat = Utils.get_field_encode_idx(sparse_dict, str(time_feat), "hour")
            distance_feat, distance_f = Utils.geodistance(sample_data["g_plng"], sample_data["g_plat"], sample_data["p_poi_lng"], sample_data["p_poi_lat"])
            #return uid_feat, q_geo_feat, q_type_feat, time_feat, poiid_feat, p_geo_feat, p_cate_feat, p_layer_feat, distance_feat, distance_f
            return q_type_feat, time_feat, p_layer_feat, distance_feat, distance_f

        def dense_norm(idx, v):
            if idx < 4:
                return v
            return math.log(v+2) / math.log(2) - 1.0

        def get_twinbert_train_union_rank(data):
            sparse_dict = get_sparse_dict(dict_file)
            res = []
            token_len = Config.TwinbertTokenLen + 14
            gbdt_feats = [] 
            gbdt_leafs = []
            gbdt_scores = []
            one_hot_offset_arr = np.array(one_hot_offset)
            for idx, line in enumerate(data):
                fields = line.strip().split("\t")
                if len(fields) < 3:
                    continue
                group_prefix = fields[1]
                sample_data = json.loads(fields[0])
                label = int(fields[3]) - 1
                # poi_info = fields[2].split("##")
                # sim_info = get_sim_feat(sample_data)
                # sample_data["sim_info"] = sim_info
                # stat_feat = _get_statistic_feat(sample_data)
                #uid_feat, q_geo_feat, q_type_feat, time_feat, poiid_feat, p_geo_feat, p_cate_feat, p_layer_feat, distance_feat, distance_f = get_id_feat(sample_data, sparse_dict)
                q_type_feat, time_feat, p_layer_feat, distance_feat, distance_f = get_id_feat(sample_data, sparse_dict)
                # 生成query侧 tokenids
                query = sample_data["g_disp_query"]
                nameaddr = sample_data["p_name_address"]
                query_token_ids, query_segment_ids, query_mask_ids = convert_query_token_ids(query, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                # query_feat_tokens = [q_geo_feat, 5, q_type_feat, 5, time_feat,5]
                # query_feat_segments = [2,2, 3,3, 4,4]
                # query_feat_masks = [1]*6
                # query_token_ids, query_segment_ids, query_mask_ids = merge_token_feat(query_token_ids, query_segment_ids, query_mask_ids, query_feat_tokens, query_feat_segments, query_feat_masks, Config.TwinbertTokenLen + 12)
                query_set = set(query.split(" "))
                query_word_len = float(len(query_set))
                name_addr_set = set(nameaddr.replace("|", " ").split(" "))
                query_set = set(query.split(" "))
                match_rat = len(query_set&name_addr_set) / query_word_len
                
                poiid_idx=sample_data["poiid_idx"]
                sample_type = sample_data["sample_type"]
                #if sample_type in ("show_hard"):
                #    continue
                vq_label = 0
                if sample_type in ("pos", "show_hard", "re_hard"):
                    vq_label = 1
                
                stat_feat = ""
                if "p_ltrfeatures" in sample_data:
                    stat_feat = parse_ltr_new_feat(sample_data, match_rat)
                if stat_feat == "":
                    stat_feat = get_statistic_feat(sample_data, sample_data["g_query_type"], distance_f, match_rat)
                dense_feat = [dense_norm(idx, float(i)) for idx, i in enumerate(stat_feat.split(";"))]
                # dense_feat = ";".join([stat_feat])
                # dense_feat = [1.0] + [float(i) for i in dense_feat.split(";")]
                # gbdt_feats.append(dense_feat)
                # if idx > 10 and idx % 50000 == 0:
                #     pred_leaf = gbm_model.predict(np.array(gbdt_feats, dtype=np.float32), pred_leaf=True, num_iteration=Config.LeafTreeLimit)
                #     tree_scores = gbm_model.predict(gbdt_feats)
                #     one_hot_pred_leaf = pred_leaf + one_hot_offset_arr + 66453
                #     gbdt_leafs.extend(one_hot_pred_leaf.tolist()) 
                #     gbdt_scores.extend(tree_scores.tolist())
                #     gbdt_feats = []

                
                
                poi_token_ids, poi_segment_ids, poi_mask_ids = convert_poi_token_ids(nameaddr, Config.TwinbertTokenLen, tokenizer, is_pad=True)
                # poi_feat_tokens = [0, 0, 0, 0, 0, 0, poi_geo_feat, 5, poi_cate_feat, 5, poi_layer_feat, 5]
                # poi_feat_segments = [0,0,0,0,0,0, 5,5, 6,6, 7,7]
                # poi_feat_masks = [0] * 6 + [1]*6
                # poi_token_ids, poi_segment_ids, poi_mask_ids = merge_token_feat(poi_token_ids, poi_segment_ids, poi_mask_ids, poi_feat_tokens, poi_feat_segments, poi_feat_masks, Config.TwinbertTokenLen + 12)
                # 生成交互模型的特征
                # nameaddr = nameaddr.split("|")
                # token_ids, segment_ids, mask_ids = Utils.get_token_id(tokenizer, query, nameaddr[0], nameaddr[1])
                # feat_tokends = [q_geo_feat, 5, q_type_feat, 5, time_feat, 5, poi_geo_feat, 5, poi_cate_feat, 5, poi_layer_feat, 5, distance_feat, 5]
                # feat_segments = [2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8]
                # feat_masks = [1]*14
                # token_ids, segment_ids, mask_ids = merge_token_feat(token_ids, segment_ids,mask_ids, feat_tokends, feat_segments, feat_masks, Config.TwinbertTokenLen + 14)
                poi_feats = [label, vq_label, poiid_idx, poi_token_ids, poi_segment_ids, poi_mask_ids, p_layer_feat, distance_feat, dense_feat, nameaddr]
                res.append([group_prefix, query, q_type_feat, time_feat,query_token_ids, query_segment_ids, query_mask_ids] + poi_feats)  
            # if len(gbdt_feats) > 0:
            #     pred_leaf = gbm_model.predict(np.array(gbdt_feats, dtype=np.float32), pred_leaf=True, num_iteration=Config.LeafTreeLimit)
            #     tree_scores = gbm_model.predict(gbdt_feats)
            #     one_hot_pred_leaf = pred_leaf + one_hot_offset_arr + 66453
            #     gbdt_leafs.extend(one_hot_pred_leaf.tolist()) 
            #     gbdt_scores.extend(tree_scores.tolist())
                
            #gbdt_leafs_all = np.concatenate(gbdt_leafs, axis=0).tolist()
            batch_size = 30000 
            batch_data = []
            new_res = []
            #gbdt_seg_ids = [9]*Config.LeafTreeLimit
            #gbdt_mask_ids = [1] * Config.LeafTreeLimit
            #for leaf, sample, score in zip(gbdt_leafs, res, gbdt_scores):
            #    sample.extend([leaf, score])
            #    new_res.append([sample[0], [sample[1:]]])
                
            return res
        
        if parse_type == "poi":
            return get_poi_data(data)
        if parse_type == "query":
            return get_query_data(data)
        if parse_type == "twinbert":
            return get_twinbert_train(data)
        if parse_type == "twinbert_l2":
            return get_twinbert_train_l2(data)
        if parse_type == "multy_feature":
            return get_multy_feature_train(data)
        if parse_type == "twinbert_new":
            return get_twinbert_train_rank(data)
        if parse_type == "twinbert_union":
            return get_twinbert_train_union_rank(data)
        return []
    def reduce_merge(x):
        group_key = x[0]
        data = sorted(x[1], key=lambda k: k[5], reverse=True)
        data = data
        pos_feat = data[0]
        res = []
        poiid_set = set()
        for feat in data:
            if feat[-7] in poiid_set:
                continue
            all_feat = [group_key]
            all_feat.extend(pos_feat)
            all_feat.extend(feat[6:])
            res.append(all_feat)
            poiid_set.add(feat[-7])
        return res
    if outformat == "poi":
        dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x))
        dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path)
        return
    if outformat in ["query", "vqgan", "twinbert", "twinbert_new", "twinbert_union", "twinbert_l2", "multy_feature"]:
        if outformat == "query":
            schema = get_struct_by_outformat(outformat)
            poiid_idx_rdd = LoadDataFromHDFS(sc, poiid_idx_path).flatMap(lambda x: [x.strip().split("\t")])
            dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x, "query")) \
                            .join(poiid_idx_rdd) \
                            .map(lambda x: out_format(x))
        elif outformat == "multy_feature":
            schema = get_struct_by_outformat(outformat)
            poiid_idx_rdd = LoadDataFromHDFS(sc, poiid_idx_path).flatMap(lambda x: [x.strip().split("\t")])
            dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x, "multy_feature")) \
                            .join(poiid_idx_rdd) \
                            .map(lambda x: out_format(x))
        elif outformat == "vqgan":
            schema = get_struct_by_outformat(outformat)
            dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x, "vqgan"))
        elif outformat == "twinbert":
            schema = get_struct_by_outformat(outformat)
            dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x, "twinbert"))
        elif outformat == "twinbert_l2":
            schema = get_struct_by_outformat(outformat)
            dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x, "twinbert_l2"))
            #dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path)
        elif outformat == "twinbert_new":
            
            schema = get_struct_by_outformat(outformat)
            dump_feature_rdd = dump_feature_rdd.mapPartitions(lambda x: _parse_sample(x, "twinbert_new"))
            #dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path)
        elif outformat == "twinbert_union":
            schema = get_struct_by_outformat(outformat)
            dump_feature_rdd = dump_feature_rdd.repartition(3000).mapPartitions(lambda x: _parse_sample(x, "twinbert_union")) 
                #.flatMap(lambda x: reduce_merge(x))
                #.reduceByKey(lambda x1, x2: x1 + x2) \
            #dump_feature_rdd.repartition(out_partitions).saveAsTextFile(save_path)
            #return
        else:
            return

        spark\
            .createDataFrame(dump_feature_rdd, schema) \
            .repartition(out_partitions) \
            .write.mode("overwrite").format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
            .save(save_path)
        # show result case
        spark\
            .read.format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
            .load(save_path + '/*') \
            .show(n=1, truncate=False)
        















def format_mtr_v1(sc, spark, click_dump_feature_rdd, order_dump_feature_rdd, out_partition, save_path, dict_file, model_file, file_type, country_code, is_sample_test, vocab_file, feature_stats):
    '''pairwise loss格式化mtr样本（初版）
    '''
    def _parse_sample(one_part_data, data_type): # format_mtr_v1
        def _load_feature_stats(feature_stats_file):
            res = {}
            for line in open(feature_stats_file):
                line = line.strip().split("\t")
                if len(line) != 5: continue
                (fkey, fmean, fstd, fmin, fmax) = line[:5]
                res[fkey] = [fmean, fstd, fmin, fmax]
            return res
        
        '''输入格式
        parse_info \t searchid||uid \t group_len \t label \t show_pos \t idx1:feat1 \t idx2:feat2 \t ...
        '''
        import numpy as np
        import lightgbm as lgbm
        import tokenization
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
        def _process_batch_data(batch_data, one_hot_offset, gbm_model): # format_mtr_v1

            '''构造特征
            '''
            def __encode_raw_feat(batch_data):
                '''处理原始信息编码
                1） 从第一个字段获得原始信息
                2）对原始特征编码
                '''
                ret = []
                for parse_info in batch_data:
                    try:
                        items = eval(parse_info)
                        # 字段强校验
                        assert len(items) >= len(Config.QueryFeatKey + Config.PoiFeatKey), "len(items) < len(Config.QueryFeatKey + Config.PoiFeatKey) %d < %d" % (len(items), len(Config.QueryFeatKey + Config.PoiFeatKey))
                        # 对原始特征编码
                        raw_feat_encode = get_id_feat(parse_info, sparse_dict, Config.QueryFeatKey + Config.PoiFeatKey)
                        ret.append(raw_feat_encode)
                    except:
                        # click数据存在部分null字段的情况
                        ret.append('')
                return ret
            
            def __get_feat_arr(batch_data):
                '''libSVM转array格式， show_pos占位idx=0
                '''
                batch_feat = []
                for data in batch_data:
                    one_feat = [0.0 for _ in range(Config.DumpFeatureCols+1)]
                    one_feat[0] = data[0]
                    for d in data[1:]:
                        idx, feat = d.split(":")
                        one_feat[int(idx)] = feat
                    batch_feat.append(one_feat)
                ret = np.array(batch_feat, dtype=np.float32)
                return ret
            
            def z_score_norm(fea_idx, raw_feat):
                res = raw_feat
                #fea_key = country_code + '_' + str(fea_idx) # 西语三国独立做归一化，效果不好， 改回统一归一化
                if(str(fea_idx) in feature_stats_dic):
                    #feature_stats_dic
                    meanx, stdx, minx, maxx = feature_stats_dic[str(fea_idx)]
                    res = (float(raw_feat) - float(meanx)) / float(stdx) if float(stdx) > 1e-6 else 0.0
                return res
            
            def __get_dense_feat_arr(batch_data): # format_mtr_v1
                '''libSVM转array格式， show_pos占位idx=0, 只保留删减后的90维特征: Config.GBDTTop90FeatList
                '''

                batch_feat = {}
                for data in batch_data:
                    one_feat = [0.0 for _ in range(len(Config.DumpFeatureCols)+1)]
                    one_feat[0] = data[0]
                    for d in data[1:]:
                        idx, feat = d.split(":")
                        one_feat[int(idx)] = float(feat)  # float
                    
                    # top90 featrues
                    batch_data.setdefault('GBDTTop90FeatList', [])
                    dense_feature = []
                    for j in range(len(Config.GBDTTop90FeatList)):
                        feat_idx = int(Config.GBDTTop90FeatList[j])
                        raw_feature = one_feat[feat_idx]
                        norm_feature = z_score_norm(feat_idx, raw_feature)
                        dense_feature.append(norm_feature)
                    batch_feat['GBDTTop90FeatList'].append(dense_feature)

                    for featGroup in Config.GBDTTopFeatDic:
                        batch_data.setdefault(featGroup, [])
                        dense_feature = []
                        featGroupList = Config.GBDTTopFeatDic[featGroup]
                        for j in range(len(featGroupList)):
                            feat_idx = int(featGroupList[j])
                            raw_feature = one_feat[feat_idx]
                            #需要对特征处理, z-score归一化
                            norm_feature = z_score_norm(feat_idx, raw_feature)
                            dense_feature.append(norm_feature)
                        batch_feat[featGroup].append(dense_feature)
                #ret = np.array(batch_feat, dtype=np.float32)
                gBBDTToop90FeatArr = np.array(batch_feat['GBDTTop90FeatList'], dtype=np.float32)
                #
                #
                #
                #
                #
                #
                #
                return gBBDTToop90FeatArr #, 
            
            # process_batch_data入口, format_mtr_v1
            ## 原始编码特征 ##
            raw_feat = [d.split('\t')[0] for d in batch_data]
            raw_feat_encode = __encode_raw_feat(raw_feat)
            ## lgbm特征 ##
            # 原特征
            traceid_uid = np.array([d.split('\t')[1] for d in batch_data]).astype(str).reshape(-1, 1)
            group_len = np.array([d.split('\t')[2] for d in batch_data]).astype(int).reshape(-1, 1)
            if data_type == "click":
                click_label = (np.array([int(d.split('\t')[3]) for d in batch_data]).astype(int) - 1).reshape(-1, 1)
                order_label = np.zeros(click_label.shape).astype(int)
            elif data_type == "order":
                order_label = (np.array([int(d.split('\t')[3]) for d in batch_data]).astype(int) - 1).reshape(-1, 1)
                click_label = order_label
            else:
                raise ValueError("data_type must in (click, order), but is %s"%data_type)
            gbdt_feat = __get_feat_arr([d.split('\t')[4:] for d in batch_data])
            #
            #
            gBDTTop90FeatList = __get_dense_feat_arr([d.split('\t')[4:] for d in batch_data])
            # 算leaf节点编码特征
            one_hot_offset_arr = np.array(one_hot_offset)
            pred_leaf = gbm_model.predict(gbdt_feat, pred_leaf=True, num_iteration=Config.LeafTreeLimit)
            one_hot_pred_leaf = pred_leaf + one_hot_offset_arr 
            # 算tree score特征
            tree_scores = gbm_model.predict(gbdt_feat)
            # query + name + address token化
            # 
            # 
            # 
            # 
            # 
            # 
            # 
            #   

            #
            #
            #
            #

            ## 转落盘格式 ##
            ret = []
            for i in range(traceid_uid.shape[0]):
                one_data = [data_type, str(traceid_uid[i][0]), int(group_len[i][0]), int(click_label[i][0]), int(order_label[i][0])]
                one_data.extend(raw_feat_encode[i])
                one_data.append(one_hot_pred_leaf[i].tolist())
                one_data.append(float(tree_scores[i]))
                one_data.append(gBDTTop90FeatList[i].tolist()) # gbdt top90特征
                #
                #
                #
                #
                #
                #
                #
                #
                # 包一层key后面用来去重
                ret.append([traceid_uid[i][0], [one_data]])
            return ret
            
        # _parse_sample入口, format_mtr_v1
        # 加载词表
        sparse_dict = get_sparse_dict(dict_file)
        feature_stats_dic = _load_feature_stats(feature_stats)
        # 加载模型
        gbm_model = lgbm.Booster(model_file=model_file)
        one_hot_offset = get_gbm_num_leaves(gbm_model)
        # 分批处理
        BATCH_SIZE = 100000
        ret = []
        read_buf = []
        for idx, l in enumerate(one_part_data):
            read_buf.append(l.strip())
            if idx % BATCH_SIZE == 0:
                ret_data = _process_batch_data(read_buf, one_hot_offset, gbm_model)
                ret.extend(ret_data)
                read_buf = []
        if len(read_buf) > 0:
            ret_data = _process_batch_data(read_buf, one_hot_offset, gbm_model)
            ret.extend(ret_data)
            
        return ret
    
    def _get_out_schema(): # format_mtr_v1
        '''拼DataFrame Schema (注意和前面特征顺序一致)
        由于是pairwise loss， 因此对于训练集来说click_label和order_label没用，只有测试的时候游泳
        '''
        struct_fields = [
            StructField("data_type", StringType(), False), 
            StructField("traceid_uid", StringType(), False),
            StructField("group_len", IntegerType(), False),
            StructField("click_label", IntegerType(), False),
            StructField("order_label", IntegerType(), False),
        ]
        query_field = get_struct_field(Config.QueryFeatKey, "")
        pos_poi_field = get_struct_field(Config.PoiFeatKey, "pos")
        neg_poi_field = get_struct_field(Config.PoiFeatKey, "neg")
        struct_fields.extend(query_field)
        struct_fields.extend(pos_poi_field)
        #struct_fields.extend(StructField("pos_token_ids", ArrayType(IntegerType(), False)))
        #struct_fields.extend(StructField("pos_segment_ids", ArrayType(IntegerType(), False)))
        #struct_fields.extend(StructField("pos_mask_ids", ArrayType(IntegerType(), False)))
        struct_fields.extend(neg_poi_field)
        #struct_fields.extend(StructField("neg_token_ids", ArrayType(IntegerType(), False)))
        #struct_fields.extend(StructField("neg_segment_ids", ArrayType(IntegerType(), False)))
        #struct_fields.extend(StructField("neg_mask_ids", ArrayType(IntegerType(), False)))
        return StructType(struct_fields), len(struct_fields)
    
    def _flatMap_train_valid(x, dump_len): # format_mtr_v1
        '''一条query'''
        ret = []
        # 拉字段
        search_uid = x[0]
        poi_list = x[1][1] if x[1][1] else x[1][0]
        # 强校验 & 分流正负样本
        pos, neg = [], []
        for poi in poi_list:
            # (1)data_type (2)traceid_uid (3)group_len (4)click_label (5)order_label <QueryFeatKey> <PoiFeatKey> (6)one_hot_pred_leaf (7)tree_scores (8-15)dense_gbdt_feat (16)token_ids (17)segment_ids (18)masked_ids
            if len(poi) != len(Config.QueryFeatKey + Config.PoiFeatKey) + 8:
                return ret
            if int(poi[3]) == 1:
                pos.append(poi)
            else:
                neg.append(poi)
        #组装pair
        for p in pos:
            for n in neg:
                one_data = []
                one_data.append(p)
                one_data.append(n[5 + len(Config.QueryFeatKey):])
                assert len(one_data) == dump_len, "len(one_data) != dump_len %d != %d" % (len(one_data), dump_len)
                ret.append(one_data)
        
        return ret
    
    def _flatMap_test(x, dump_len): #format_mtr_v1
        '''一条query'''
        ret = []
        # 拉字段
        search_uid = x[0]
        poi_list = x[1]
        # 强校验 & 分流正负样本
        pos, neg = [], []
        for poi in poi_list:
            # (1)data_type (2)traceid_uid (3)group_len (4)click_label (5)order_label <QueryFeatKey> <PoiFeatKey> (6)one_hot_pred_leaf (7)tree_scores (8-15)dense_gbdt_feat (16)token_ids (17)segment_ids (18)masked_ids
            if len(poi) != len(Config.QueryFeatKey + Config.PoiFeatKey) + 8:
                return ret
            if int(poi[3]) == 1:
                pos.append(poi)
            else:
                neg.append(poi)
        #组装pair
        for p in pos + neg:
            one_data = []
            one_data.append(p)
            one_data.append(p[5 + len(Config.QueryFeatKey):])
            assert len(one_data) == dump_len, "len(one_data) != dump_len %d != %d" % (len(one_data), dump_len)
            ret.append(one_data)
        
        return ret
    
    #format_mtr_v1 入口
    if file_type in ('train', 'valid'):
        # 生产训练集和验证集
        # 算gbdt叶子编码和打分
        click_encode_rdd = click_dump_feature_rdd \
            .mapPartitions(lambda x: _parse_sample(x, "click")) \
            .reduceByKey(lambda x1, x2: x1 + x2)
        
        order_encode_rdd = order_dump_feature_rdd \
            .mapPartitions(lambda x: _parse_sample(x, "order")) \
            .reduceByKey(lambda x1, x2: x1 + x2)
        
        # 去重（click包含全部order信息）
        schema, dump_len = _get_out_schema()
        print('format_mtr_v1 schema:', schema)
        eval_rdd = click_encode_rdd\
            .leftOuterJoin(order_encode_rdd) 
        print('eval_rdd count:', eval_rdd.count())
        if is_sample_test != 'all':
            case_num = int(is_sample_test)
            total_num  = eval_rdd.count()
            print("")
            print("")
            eval_rdd = eval_rdd.sample(withReplacement=False, fraction=1.0 * case_num / total_num, seed=12357)
            rdd = eval_rdd.flatMap(lambda x: _flatMap_train_valid(x, dump_len))
        else:
            rdd = eval_rdd.flatMap(lambda x: _flatMap_train_valid(x, dump_len))
    else:
        # 生产测试集
        schema, dump_len = _get_out_schema()
        #
        eval_rdd = order_dump_feature_rdd \
            .mapPartitions(lambda x: _parse_sample(x, "order")) \
            .reduceByKey(lambda x1, x2: x1 + x2)
        
        if is_sample_test != 'all':
            case_num = int(is_sample_test)
            total_num  = eval_rdd.count()
            print("")
            print("")
            eval_rdd = eval_rdd.sample(withReplacement=False, fraction=1.0 * case_num / total_num, seed=12357)
            rdd = eval_rdd.flatMap(lambda x: _flatMap_test(x, dump_len))
            print("")

    # format_mtr_v1
    # 转tfrecords格式落盘
    print('final rdd :', rdd.take(2))
    spark \
        .createDataFrame(rdd, schema) \
        .repartition(out_partition) \
        .write.mode("overwrite").format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .save(save_path)
    
    #验证落盘格式
    df = spark \
        .read.format("tfrecords").option("recordType", "Example").option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .load(save_path + '/*')
    #打印前20行
    df.show()
    # 统计数量
    df.agg(countDistinct(col("traceid_uid")).alias("traceid_uid")).show()
