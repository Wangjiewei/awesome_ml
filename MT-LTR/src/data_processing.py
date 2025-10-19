# encoding=utf8
from output_data_operator import format_deepfm, format_mtr_pointwise, format_mtr_v1, format_mtr_v2
from utils import LoadDataFromHive, Utils
from pyspark.sql import SparkSession
import parse_sample_operator
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

spark = SparkSession \
    .builder \
    .appName("search_sample_product") \
    .master("yarn") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "12g") \
    .config("spark.yarn.executor.memoryOverhead", "1g") \
    .config("spark.dynamicAllocation.maxExecutors", 300) \
    .config("spark.dynamicAllocation.minExecutors", 100) \
    .config("spark.executor.cores", 1) \
    .config("spark.port.maxRetries", 100) \
    .enableHiveSupport() \
    .getOrCreate()
sc = spark.sparkContext
sc.pythonExec = spark.conf.get("spark.yarn.appMasterEnv.PYSPARK_PYTHON")


def test_parse_line(x):
    res = ""
    data = x.split("\t")[4:]
    flag = False
    for d in data[1:]:
        temp = d.split(":")
        if(len(temp) != 2):
            res = "\t".join([str(len(temp)), d, x])
            break
    return res


if __name__ == "__main__":

    start_yyyymmdd = sys.argv[1]
    end_yyyymmdd = sys.argv[2]
    country_code = sys.argv[3]
    dump_feature_log_root_path = sys.argv[4]
    hdfs_output_root_path = sys.argv[5]
    output_partitions = int(sys.argv[6])
    output_format = sys.argv[7]
    dic_file = None
    model_file = None
    click_featureAll_hdfs_path = None
    file_type = None
    is_sample_test = None
    if output_format == "deepfm":
        dic_file = sys.argv[8]
        model_file = sys.argv[9]
    if output_format == "mtr" or output_format == "click_featureAll":
        dic_file = sys.argv[8]
        model_file = sys.argv[9]
        click_featureAll_hdfs_path = sys.argv[10]
        file_type = sys.argv[11]
    if output_format == "mtr":
        is_sample_test = sys.argv[12]
        vocab_file = sys.argv[13]
        feature_stats = sys.argv[14]

    ## lgbm ##
    if output_format == "gbdt":
        # 特征宽表解析
        dump_feature_rdd = LoadDataFromHive(spark, start_yyyymmdd, end_yyyymmdd, country_code)
        dump_feature_rdd = dump_feature_rdd \
            .flatMap(lambda x: parse_sample_operator.parse_sample_operator(x, country_code)) \
            .reduceByKey(lambda x, y: x) \
            .map(lambda x: x[1]) \
            .repartition(output_partitions) \
            .saveAsTextFile(hdfs_output_root_path)
    
    ## deepfm ##
    elif output_format == "deepfm":
        input_hdfs_path = os.path.join(dump_feature_log_root_path, country_code, "feature_dump", start_yyyymmdd + "_" + end_yyyymmdd)
