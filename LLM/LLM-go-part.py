# coding: gbk
import ollama
import numpy as np
import pandas as pd
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
from sklearn import metrics
import os
import re
from datetime import datetime
MAX_FEATURES = 3  # 每个类别提取3个特征
CATEGORIES = ['BP', 'MF', 'CC']  # 三个功能类别

# def is_valid_output(feature):
#     """验证特征有效性"""
#     return (
#         # len(feature) >= 3 and
#         # not feature.startswith('http') and
#         # any(c.isalpha() for c in feature)
#     )

def process_model_output(content, gene_name):
    """处理大模型输出并提取三类特征"""
    # 特征类别
    CATEGORIES = ['BP', 'MF', 'CC']
    MAX_FEATURES = 3
    
    # 初始化结果字典
    features = {
        'BP': ['none'] * MAX_FEATURES,
        'MF': ['none'] * MAX_FEATURES,
        'CC': ['none'] * MAX_FEATURES
    }
    
    try:
        # 使用正则表达式匹配三个类别的内容
        for category in CATEGORIES:
            pattern = r'</{}>(.*?)</{}>'.format(category, category)
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                # 清理内容块
                cleaned_block = re.sub(r'[\n\t]+', ' ', match.group(1).strip())
                
                # 分割特征项
                raw_features = re.split(r'[,;]', cleaned_block)
                
                # 处理每个特征
                valid_features = []
                for feat in raw_features:
                    feat = re.sub(r'^\d+[\.\)]?\s*', '', feat.strip())
                    if feat and is_valid_output(feat):
                        valid_features.append(feat.lower())
                
                # 填充特征列表
                features[category] = valid_features[:MAX_FEATURES]
                if len(features[category]) < MAX_FEATURES:
                    features[category] += ['none'] * (MAX_FEATURES - len(features[category]))

    except Exception as e:
        print(f"处理{gene_name}输出时出错: {str(e)}")
        return None
        
    return features

def is_valid_output(feat):
    return len(feat) > 4 and not re.search(r'[$]', feat)

EPOCH = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prompt = open(".\LLM\go_prompt.txt", "r",encoding='gbk').read()

cancer_name_txt = pd.read_csv("/home/yuantao/code/LLM/txt/data/cancer.txt", header=None).values

sentance = pd.read_csv(r'/home/yuantao/code/LLM/GO词汇获取/string_test.csv')
sentance.columns = ['Ensembl_ID','Gene_Symbol','All_Description','Has_Description']


cancer_names = [ 'pan-cancer']
modelname = 'gemma2'


for cancer_name in cancer_names:
    cancer_name = cancer_name.upper()
    print(f"Processing {cancer_name}...")
    cancer_name_prompt = cancer_name
    for row in cancer_name_txt:
        if row[0].startswith(cancer_name_prompt):
            cancer_name_prompt = row[0]
            cancer_name_prompt = cancer_name_prompt.replace("\t"
            , "_")
            break
    new_prompt = prompt.replace('Cancer_name', cancer_name_prompt)
    gene_name_list = open(r'/home/yuantao/code/LLM/txt/data/string_node_names.txt','r',encoding='utf-8').read().split('\n')


    output_dir = '/home/yuantao/code/LLM/csv/分类型输出/' + modelname + '/'
    output_file = os.path.join(output_dir, f"{cancer_name}_string_go_features.csv")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            header = ['Gene_ID', 'Gene_Symbol']
            for cat in CATEGORIES:
                header += [f"{cat}_Feature{i+1}" for i in range(MAX_FEATURES)]
            f.write(','.join(header) + '\n')
    
    f1 = open(output_file, "a")
    
    if os.path.exists(output_file):

        with open(output_file, 'r') as f:
            lines = f.readlines()
            if len(lines)==1:
                lenth = 0
            elif len(lines)<len(gene_name_list):
                lenth = len(lines)
                last_id = lines[-1].split(',')[0]
                # print('last_id:', last_id)
                last_name = lines[-1].split(',')[1]
                last_gene = str(last_id) + '\t' + str(last_name)
                lenth = gene_name_list.index(last_gene) + 1
                # gene_name_list = gene_name_list[len(lines):]
            else:
                print('All done')
                continue
        print('begin from:', gene_name_list[lenth])
    
        for idx, gene_entry in enumerate(gene_name_list):
            if idx < lenth:
                continue
                
            gene_id, gene_symbol = gene_entry.split('\t', 1)
            retry_count = 0
            success = False
            
            while retry_count < 3 and not success:
                try:
                    # 生成prompt
                    new_prompt1 = new_prompt.replace('Gene_name', gene_symbol).replace(
                        'go_description', str(sentance['All_Description'][idx])
                    )
                    # print(new_prompt1)
                    
                    # 获取模型响应
                    response = ollama.generate(model=modelname, prompt=new_prompt1)
                    content = response['response']
                    # print(f"模型响应: {content}")
                    
                    # 处理输出
                    feature_dict = process_model_output(content, gene_symbol)
                    
                    if feature_dict:
                        # 构建CSV行
                        row_data = [gene_id, gene_symbol]
                        for cat in CATEGORIES:
                            row_data += feature_dict[cat][:MAX_FEATURES]
                        
                        # 写入文件
                        # with open(output_file, 'a') as f:
                        #     f.write(','.join(map(str, row_data)) + '\n')
                        f1.write(','.join(map(str, row_data)) + '\n')
                            
                        print(f"{gene_symbol},{feature_dict}")
                        success = True
                    else:
                        print(f"第{retry_count+1}次重试: {gene_symbol}")
                        retry_count += 1
                        
                except Exception as e:
                    print(f"处理异常: {str(e)}")
                    retry_count += 1
                    
            if not success:
                print(f"无法获取有效特征: {gene_symbol}")
                # 记录失败案例
                with open('error_log.txt', 'a') as f:
                    f.write(f"{gene_id},{gene_symbol}\t{content}\n")

            # 定期刷新缓冲区
            if idx % 20 == 0:
                f1.flush()
                print(f"已处理 {idx} 个基因，当前基因: {gene_symbol}")

    f1.close()
