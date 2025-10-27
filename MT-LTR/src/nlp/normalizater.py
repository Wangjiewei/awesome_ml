# -*- coding:utf-8 -*-
'''
 author: Your Name
 summary: Text normalization utilities for NLP tasks
'''
import sys
import os
import json
import re
import logging
from trad2simp_dict import trad_simp_dict
reload(sys)
sys.setdefaultencoding('utf-8')

#
SPECIAL_SYMBOL = ["","","_","-"]
SPECIAL_SYMBOL_CO = []

IS_DEBUG = False
#

def absPath(file_path):
    return os.path.normpath(os.path.join(os.getcwd(),os.path.dirname(__file__), file_path))

# 加载重音符映射表
baxi_norm_char_dict = {}
norm_char_lines = open('norm_char.txt', "r").readlines()


# 加载数字归一映射表
baxi_num_norm_dict = {}
num_norm_lines = open('num_norm.txt', "r").readlines()

for line in norm_char_lines:
    key, value = line.strip().split("\t")
    key = key.decode("utf-8")
    baxi_num_norm_dict[key] = value

for line in num_norm_lines:
    key, value = line.strip().split("\t")
    key = key.decode("utf-8")
    baxi_norm_char_dict[key] = value

# 加载停用词表
baxi_stop_words_list = []
stop_words_lines = open('stop_words.txt', "r").readlines()
for line in stop_words_lines:
    baxi_stop_words_list.append(line.strip())


# 巴西葡语归一化二期 
# 
pattern = re.compile(r'^[0-9]*$')

# （判断是否字符是否为数字）对于纯整数，可以使用a.isdigit()函数
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
# 加载巴西（葡语数字-阿拉伯数字）对应词表
PT_NUM_DICT = {}
pt_num_dict_lines = open('pt_num.txt', "r").readlines()
for line in pt_num_dict_lines:
    num, pt_char = line.strip().split("\t")
    PT_NUM_DICT[pt_char] = str(num)


class Normalizate(object):
    def __init__(self):
        global baxi_stop_words_list
        sen = '\t'.join(baxi_stop_words_list)
        sen = self.baxi_special_mark_process(sen)
        baxi_stop_words_list = sen.split('\t')
        self.baxi_stop_words_set = set(baxi_stop_words_list)

    '''
    def is_alpha_digti(self, word):
        return word.isalpha() or word.isdigit()
    '''

    def do_normalizate(self, sen):
        if sen == "":
            return ""
        if not isinstance(sen, unicode):
            sen = unicode(sen,"utf-8")
        sen = self.SBC2DBC(sen)
        sen = self.traditional2simple(sen)
        sen = sen.lower()
        words = self.get_words(sen)
        res_words = []
        i = 0
        while i < len(words):
            if words[i] in SPECIAL_SYMBOL:
                while i < len(words) and words[i] in SPECIAL_SYMBOL:
                    i += 1
                res_words += [" "]
            else:
                res_words += [words[i]]
                i += 1
        res_str = ''.join(res_words)
        res = re.sub(r"\s{2,}", " ", res_str)
        return res.strip(' ')
    
    def do_normalizate_baxi(self, sen):
        pass










































    # 巴西葡语归一化二期
    def do_normalizate_baxi_v2(self, sen):
        pass












































































    # 哥伦比亚西语归一化
    def do_normalizate_co(self, sen):
        pass












































    def remove_duplicate_symbol(self, string):
        words = string.split()
        if len(words) < 2:
            return ' '.join(words)
        
        res_words = []
        res_words.append(words[0])
        for i in range(1, len(words)):
            pre_word = words[i-1]
            word = words[i]
            if word in ['-', '#'] and word == pre_word:
                continue
            else:
                res_words.append(word)
        return ' '.join(res_words)
    
    def replace_number_symbol(self, words):
        if len(words) < 2:
            return words
        res_words = []
        for idx in range(len(words)):
            if words[idx] in ['n','no','numero'] and idx < len(words)-1 and self.is_str_number(words[idx+1]):
                res_words.append('#')
            else:
                res_words.append(words[idx])
        return res_words
    
    def omit_sur_av(self, words):
        res_words = words
        if len(words) >= 2 and words[-1] == 'sur' and self.is_str_number(words[-2]):
            res_words = res_words[:-1]
        if len(words) >= 2 and words[0] == 'av' and 'calle' in words:
            res_words = res_words[1:]
        return res_words
    
    def omit_bad_blankspace(self, words):
        res_words = []
        for idx in range(len(words)):
            word = words[idx]
            if idx == 0:
                res_words.append(word)
                continue
            pre_word = words[idx-1]
            if len(word) == 1 and self.is_single_alpha(word) and self.is_str_number(pre_word):
                res_words[-1] += word  # 数字 + 单个字母
            elif self.is_str_number(word) and pre_word == '#':
                res_words[-1] += word  # ‘#’ + 数字
            elif self.is_str_number(word) and pre_word == '-':
                res_words[-1] += word  # ‘-’ + 数字
            elif word == '-' and self.is_str_number(pre_word):
                res_words[-1] += word  # 数字 + ‘-’
            elif word == '-' and (len(pre_word) == 1 and self.is_single_alpha(pre_word)):
                res_words[-1] += word # 单个字母 + ‘-’
            else:
                res_words.append(word)
        return res_words
    
    # 这些策略从DA的CO归一化而来，具体可能之后会调整


























    def strategy_co(self, sen):
        pass











    def remove_stop_words_baxi(self, sen):
        pass






















    # co的停用词表暂时用巴西的
    def remove_stop_words_co(self, sen):
        pass























    def baxi_special_mark_process(self, sen):
        pass















    # 巴西数字归一
    def baxi_num_norm(self, sen):
        pass













    # @brief:繁体转简体
    def traditional2simple(self, sen):
        res = ""
        for ch_word in line:
            if ch_word in trad_simp_dict:
                res += trad_simp_dict[ch_word]
            else:
                res += ch_word
        return res
    
    # @brief:全角转半角
    def SBC2DBC(self, ustring):
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xFEE0
                #转完之后不是半角字符返回原来的字符
                if inside_code < 0x0020 or inside_code > 0x7E:
                    rstring += uchar
                else:
                    rstring += unichr(inside_code)
        return rstring
    

    # @brief:判断一个unicode是否是汉字
    def is_chinese(self, uchar):
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        else:
            return False
        

    # @brief:获判断一个unicode是否是数字
    def is_number(self, uchar):
        if uchar >= u'\u0030' and uchar <= u'\u0039':
            return True
        else:
            return False
        
    # 判断一个字符是否是数字
    def is_str_number(self, string):
        for c in string:
            if not self.is_number(c):
                return False
        return True
    
    # @brief:判断一个unicode是否是英文字母
    def is_alphabet(self, uchar):
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
            return True
        else:
            return False
        
    def is_alphabet_baxi(self, uchar):
        if self.is_alphabet(uchar) or uchar in baxi_norm_char_dict:
            return True
        return False
    
    def is_single_alpha(self, string):
        if len(string) != 1:
            return False
        return self.is_alphabet(string)
    
    # @brief: 判断一个unicode既不是英文字母、汉字，也不是数字
    def is_other(self, uchar):
        if not self.is_chinese(uchar) and not self.is_alphabet(uchar) and not self.is_number(uchar):
            return True
        else:
            return False
        
    
    # @brief: 判断一个unicode既不是英文字母、汉字，也不是数字
    def is_other_baxi(self, uchar):
        if not self.is_chinese(uchar) and not self.is_alphabet_baxi(uchar) and not self.is_number(uchar):
            return True
        else:
            return False
        
    # @brief: 获取字符串中的单个汉字，连续的英文， 连续数字，特殊符号
    def get_words(self ,s):
        words = []
        c_s = ''
        i = 0
        while i < len(s):
            if self.is_alphabet(s[i]):
                c_s = ''
                while i < len(s) and self.is_alphabet(s[i]):
                    c_s += s[i]
                    i += 1
                words.append(c_s)
                continue
            elif self.is_number(s[i]):
                c_s = ''
                while i < len(s) and self.is_number(s[i]):
                    c_s += s[i]
                    i += 1
                words.append(c_s)
                continue
            elif self.is_other(s[i]):
                c_s = ''
                words.append(s[i])
                i += 1
                continue
            elif self.is_chinese(s[i]):
                c_s = ''
                words.append(s[i])
                i += 1
                continue
        return words
    
    def get_words_baxi(self, s):
        pass






























    def remove_special_symbols_co(self, string, replace=' '):
        pass

















    def get_words_co(self, s):
        pass





