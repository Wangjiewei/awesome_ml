# encoding=utf8
from ast import dump
import sys
import os
import json
import random
import datetime
from config import Config
from pyspark.sql import SparkSession
from datetime import timedelta
from collections import defaultdict
#sys.path.append("operators")
#sys.path.append("common")
import parse_sample_operator, add_new_feat_operator
from output_data_operator import format_gbdt, format_deepfm, format_bert, format_twinbert
from utils import LoadDataFromHDFS, Utils
reload(sys)
sys.setdefaultencoding('utf-8')

spark = SparkSession \
    .builder \
    .appName("search_sample_product") \
    .master("yarn") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.yarn.executor.memoryOverhead", "1g") \
    .config("spark.dynamicAllocation.maxExecutors", 150) \
    .config("spark.dynamicAllocation.minExecutors", 100) \
    .config("spark.executor.cores", 2) \
    .config("spark.port.maxRetries", 100) \
    .enableHiveSupport() \
    .getOrCreate()

sc = spark.sparkContext
sc.pythonExec = spark.conf.get('spark.yarn.appMasterEnv.PYSPARK_PYTHON')

start_yyyymmdd = sys.argv[1]
end_yyyymmdd = sys.argv[2]
country_code = sys.argv[3]
dump_feature_log_root_path = sys.argv[4]
hdfs_output_root_path = sys.argv[5]
output_partitions = int(sys.argv[6])
output_format = sys.argv[7]
dict_file = None
model_file=None
if output_format == "deepfm" or output_format == "bert":
    dict_file = sys.argv[8]
    model_file = sys.argv[9]

# 直接读取指定目录数据
if output_format in ("twinbert", "poi", "query", "multy_feature"):
    dump_feature_rdd = LoadDataFromHDFS(sc, dump_feature_log_root_path)
    print(output_format)
else:
    #特征宽表数据
    dump_feature_rdd = LoadDataFromHive(spark, start_yyyymmdd, end_yyyymmdd, country_code)
    dump_feature_rdd = dump_feature_rdd.flatMap(lambda x: parse_sample_operator.parse_sample_operator(x))
    dump_feature_rdd.repartition(10).saveAsTextFile(hdfs_output_root_path)
# 执行特征拼接算子
for operator, params in Config.AddNewFeatMap.items():
   if params["is_exec"]:
       dump_feature_rdd = eval("add_new_feat_operator."+operator)(sc, dump_feature_rdd, params)
## 格式化输出
if output_format == "gbdt":
    format_gbdt(dump_feature_rdd, output_partitions, hdfs_output_root_path)
if output_format == "deepfm":
    format_deepfm(sc, spark, dump_feature_rdd, output_partitions, hdfs_output_root_path, dict_file, model_file)

if output_format == "bert":
    format_bert(sc, spark, dump_feature_rdd, output_partitions, hdfs_output_root_path, dict_file, model_file)
if output_format in ("twinbert", "poi", "query","multy_feature"):
    poiid_idx_path = sys.argv[8]
    vocab_file = sys.argv[9]
    dict_file = None
    dump_feature_log_root_path_2 = sys.argv[11]
    print(hdfs_output_root_path)
    format_twinbert(sc, spark, dump_feature_rdd, output_partitions, hdfs_output_root_path, output_format, poiid_idx_path, vocab_file, dict_file)
    if output_format in( "query", "multy_feature" ):
        hdfs_output_root_path_2 = hdfs_output_root_path + '_test'
        output_partitions = 500
        dump_feature_rdd_2 = LoadDataFromHDFS(sc, dump_feature_log_root_path_2)
        print(dump_feature_log_root_path_2, dump_feature_rdd_2.count())
        print(hdfs_output_root_path_2)
        format_twinbert(sc, spark, dump_feature_rdd_2, output_partitions, hdfs_output_root_path_2, output_format, poiid_idx_path, vocab_file, dict_file)
