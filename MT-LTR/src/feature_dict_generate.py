# encoding=utf8
from pyspark.sql import SparkSession
from utils import LoadDataFromHive, Utils
from config import Config
import parse_sample_operator
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

spark = SparkSession \
    .builder \
    .appName("feat_dict_generate") \
    .master("yarn") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.dynamicAllocation.maxExecutors", 200) \
    .config("spark.dynamicAllocation.minExecutors", 100) \
    .config("spark.executor.cores", 2) \
    .config("spark.port.maxRetries", 100) \
    .enableHiveSupport() \
    .getOrCreate()
sc = spark.sparkContext
sc.pythonExec = spark.conf.get("spark.yarn.appMasterEnv.PYSPARK_PYTHON")

def process_one_partition(partition_data):
    def _get_id_feat_dict(sample_info):
        import id_convert_operator
        feat_dict = []
        for feat_conf in Config.QueryFeatKey + Config.PoiFeatKey:
            sample_key = feat_conf['sample_key']
            if "dict_func" in feat_conf:
                query_feat = eval("id_convert_operator." + feat_conf['dict_func'])(sample_info, feat_conf)
            else:
                # query_feat格式： [编码类型||编码值，频次]
                query_feat = [ [ "||".join([feat_conf["dict_key"], str(sample_info[sample_key])]), 1] ]
            feat_dict.extend(query_feat)
        
        return feat_dict

    def process_one_part(partition_data):
        ret = []
        # 特征编码
        parse_info = json.loads(partition_data.split("\t")[0])
        ret_data = _get_id_feat_dict(parse_info)
        ret.extend(ret_data)
        # 补充UNK
        dict_keys = []
        for feat_conf in Config.QueryFeatKey + Config.PoiFeatKey:
            dict_key = feat_conf["dict_key"]
            if dict_key in dict_keys:
                continue
            ret.append( ["||".join([dict_key, "UNK"]), 1] ) # ret.apppend(["%s||%s" % (dict_key, "UNK"), 1])

        return ret
    
    return process_one_part(partition_data)

def run(spark, start_yyyymmdd, end_yyyymmdd, country_code):
    '''从特征宽表捞字段（默认下游是订单表）
        格式：parse_info \t searchid||uid \t group_len \t label \t show_pos \t idx1:feat1 \t idx2:feat2 \t ...
    '''
    dump_feature_rdd = LoadDataFromHive(spark, start_yyyymmdd, end_yyyymmdd, country_code)
    ret = dump_feature_rdd \
        .flatMap(lambda x: parse_sample_operator.parse_sample_operator(x, country_code)) \
        .reduceByKey(lambda x, y: x) \
        .map(lambda x: x[1]) \
        .flatMap(lambda x: process_one_partition(x))
    
    return ret

if __name__ == "__main__":
    
    start_yyyymmdd = sys.argv[1]
    end_yyyymmdd = sys.argv[2]
    country_code_list = sys.argv[3].split(",")
    dump_feature_log_root_path = sys.argv[4]
    hdfs_output_root_path = sys.argv[5]
    output_partitions = int(sys.argv[6])

    #编码字段配置
    filter_kv = { feat_conf['dict_key']: feat_conf["filter_th"] for feat_conf in Config.QueryFeatKey + Config.PoiFeatKey if "filter_th" in feat_conf }

    # 国家码校验
    if len(country_code_list) <= 0:
        raise ValueError("country_code_list is empty!")
    
    # 编码
    ret = run(spark, start_yyyymmdd, end_yyyymmdd, country_code_list[0])
    for country_code in country_code_list[1:]:
        dump_feature_rdd = run(spark, start_yyyymmdd, end_yyyymmdd, country_code)
        ret = ret.union(dump_feature_rdd)

    # 频次过滤 & 落盘
    ret.reduceByKey(lambda x, y: x + y) \
        .flatMap(lambda x: [ [x[0], x[1]] ]) \
        .filter(lambda x: (x[0].split("||")[0] not in filter_kv) or \
                 ( (x[0].split("||")[0] in filter_kv) and (x[1] > filter_kv[x[0].split("||")[0]]) ) ) \
        .flatMap(lambda x: [ "\t".join([x[0], str(x[1])]) ]) \
        .repartition(output_partitions) \
        .saveAsTextFile(hdfs_output_root_path)
    
    