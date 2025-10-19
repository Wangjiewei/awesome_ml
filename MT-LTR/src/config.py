#coding:utf-8


class Config:
    '''
    公共配置
    '''
    DumpFeatureCols = 529
    LeafTreeLimit = 300
    GEO_PRECISION = 6
    ParseL1Sample = False
    BertTokenLen = 60
    TwinbertTokenLen = 32

    #特征拼接算子
    AddNewFeatMap = {


































    }

    '''
    配置需要在原始样本中解析的字段
    sample_key: 原始特征中字段名｜｜解析算子生成的key，如果是多个以逗号分隔
    parse_func: 针对一些特殊字段，可以设置单独的解析算子，默认不用配
    default_val: 当取不到字段的默认值，默认为空
    '''
    QuerySampleInfo = [
        {"sample_key": "g_disp_query", "parse_func":"parse_query"},
        {"sample_key": "g_disp_area"},
        {"sample_key": "g_uid"},
        {"sample_key": "g_plng"},
        {"sample_key": "g_plat"},
        {"sample_key": "g_timestamp"},
        {"sample_key": "g_query_type"},
        {"sample_key": "q_geohash", "parse_func":"parse_query_geohash"},
        {"sample_key": "component_terms, component_ids", "parse_func":"parse_da_result"}
    ]

    PoiSmapleInfo = [
        {"sample_key": "p_poi_id"},
        {"sample_key": "p_name_address", "parse_func":"parse_name_address"},
        {"sample_key": "p_poi_lat"},
        {"sample_key": "p_poi_lng"},
        {"sample_key": "p_category"},
        {"sample_key": "p_geometry"},
        {"sample_key": "p_brand"},
        {"sample_key": "p_show_pos"},
        {"sample_key": "p_layer", "default_val": 0},
        {"sample_key": "p_geohash", "parse_func":"parse_poi_geohash"}
    ]

    '''
    wiki: 
    sample输出结果的key，下游复用。分为QueryFeat和PoiFeat
    conver_id_func: id生成函数
    dict_key: 构造深度模型映射词典字段对应的前缀
    dict_func: 生成字典时，字段处理算子
    out_type: 特征的数据类型
    pad_size: array类型的特征需要拉齐的长度 
    '''
    QueryFeatKey = [
        {'sample_key':'g_disp_query', 'dict_func':'cv_tri_dict', 'dict_key':'trigram', 'convert_id_func':'cv_tri_id', 'pad_size':100, 'out_type':'array'},
        {'sample_key':'component_ids', 'dict_key':'comp_label_id', 'dict_func':'cv_component_id_dict', 'convert_id_func':'cv_component_id', 'pad_size':10, 'out_type':'array'},
        {'sample_key': 'g_disp_area', 'dict_key':'cityid', 'out_type':'int'},
        {'sample_key':'g_uid', 'dict_key':'uid', 'filter_th': 0, 'out_type': 'int'} ,



    ]
    #
    #
    
    PoiFeatKey = [
        {'sample_key':'p_name_address', 'dict_func':'cv_tri_dict', 'dict_key':'trigram', 'convert_id_func':'cv_tri_id', 'pad_size':150, 'out_type':'array'},
        {'smaple_key':'p_category', 'dict_key':'category', 'dict_func':'cv_category_dict', 'convert_id_func':'cv_category_id', 'out_type': 'array', 'pad_size': 10},
        {'sample_key':'p_poi_id', 'dict_key':'poiid', 'filter_th': 0, 'out_type': 'int'} ,
        {'sample_key': 'p_geohash', 'dict_key':'geohash', 'out_type':'int'},
        {'sample_key': 'p_layer', 'dict_key': 'layer', 'out_type':'int'}
    ]
    #
    #
    #

    # bert模型训练时需要生成的特征，配置方式同上
    BertQueryFeatKey = [
        {'sample_key':'g_uid', 'dict_key':'uid', 'filter_th': 1, 'out_type': 'int'} 
    ]

    # sample输出中poi相关的key， 下游复用， 格式同上
    BertPoiFeatKey = [
        {'sample_key':'p_poi_id', 'dict_key':'poiid', 'filter_th': 3, 'out_type': 'int'}
    ]
    # gbdt 重要度top的特征，均值，最小值，最大值
    GbdtFeatMap = {
        "32": [1.8284, -1.0, 5.0],


































    }
    DenseFeatMap = {
        0: [16.0, 0.0, 205.0],


































































































































































































































    }
    GbdtUseFeatIndex = {}
    count = 1
    for i in range(1, 37):
        if i in (16, 25,27, 29,30, 33,34,35,36):
            continue
        GbdtUseFeatIndex[i] = count
        count += 1
    GbdtUseFeatIndex[i] = count
    count += 1
    for i in range(64,69):
        GbdtUseFeatIndex[i] = count
        count += 1




    





    GBDTTop90FeatList = ['66']

    #
    #
    #
    #
    ##
    #
    #
    GBDTTopFeatDic = {
        'GBDTRawFeatList' : [],
        'GBDTDestPidCity': [],
        'GBDTDestQueryCity': [],
        'GBDTDestQueryGeo': [],
        'GBDTDestPidQueryCity': [],
        'GBDTStartQueryGeo': [],
        'GBDTACDestQueryGeo': []
    }

    




