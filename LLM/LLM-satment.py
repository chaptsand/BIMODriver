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
MAX_FEATURES = 1
CATEGORIES = ['self_statement', 'neighbor_statement', 'together_statement']

def get_neighbor_descriptions(node_idx, data, sentance, max_neighbors=6):
    edge_index = data.edge_index.cpu().numpy()
    sources = edge_index[0]
    targets = edge_index[1]
    neighbors = []
    neighbors.extend(sources[targets == node_idx].tolist())
    neighbors.extend(targets[sources == node_idx].tolist())
    
    # 去重并排序以保证顺序一致性
    unique_neighbors = sorted(list(set(neighbors)))
    
    # 分两个阶段收集：有描述的 + 无描述的
    desc_candidates = []
    no_desc_candidates = []
    
    for n in unique_neighbors:
        gene_symbol = data.node_names[n][1]
        node_data = sentance[sentance['Gene_Symbol'] == gene_symbol]
        
        if not node_data.empty:
            all_desc = node_data['All_Description'].iloc[0]
            if node_data['Has_Description'].iloc[0]:
                desc_candidates.append((gene_symbol, all_desc))
            else:
                no_desc_candidates.append((gene_symbol, all_desc))
    
    # 构建描述列表
    descriptions = []
    
    # 第一阶段：添加有描述的邻居
    for gene_symbol, desc in desc_candidates[:max_neighbors]:
        # print(f"with description: {gene_symbol}, description: {desc}")
        descriptions.append(f"{gene_symbol}: {desc}")
    
    # 第二阶段：如果不足则补充无描述的邻居
    remaining = max_neighbors - len(descriptions)
    if remaining > 0:
        for gene_symbol in no_desc_candidates[:remaining]:
            # print(f"with no description: {gene_symbol}")
            descriptions.append(f"{gene_symbol}")
    
    # 处理无邻居的情况
    return "\n".join(descriptions) if descriptions else "No neighboring gene information available"

def process_model_output(content, gene_name):
    """处理大模型输出并提取三类描述性语句（简化版）"""
    CATEGORY_TAGS = {
        'self_statement': r'</self_statement>(.*?)</self_statement>',
        'neighbor_statement': r'</neighbor_statement>(.*?)</neighbor_statement>',
        'together_statement': r'</together_statement>(.*?)</together_statement>'
    }
    
    statements = {cat: 'none' for cat in CATEGORY_TAGS.keys()}

    try:
        # 统一处理换行符和空白字符
        normalized_content = content.replace('\r\n', '\n').replace('\r', '\n').strip()
        
        # 直接匹配各分类标签
        for category, pattern in CATEGORY_TAGS.items():
            match = re.search(pattern, normalized_content, re.DOTALL)
            
            if match:
                raw_text = match.group(1)
                # 增强清洗流程
                cleaned_text = re.sub(r'[\t\xa0]+', ' ', raw_text)    # 替换特殊空白
                cleaned_text = re.sub(r'\n{2,}', '\n', cleaned_text)   # 合并多余换行
                cleaned_text = re.sub(r'[^\w\s,.;:()\-/]', '', cleaned_text)  # 过滤特殊符号
                cleaned_text = cleaned_text.strip().lower()
                
                # 有效性验证（保留空行检测）
                if 20 <= len(cleaned_text) <= 1000:
                    statements[category] = cleaned_text
                else:
                    print(f"内容长度异常: {len(cleaned_text)}字符")
            else:
                print(f"未检测到{category}标签")

    except Exception as e:
        print(f"处理{gene_name}时发生异常: {str(e)}")
        return None
        
    return statements

EPOCH = 100

data = Data.from_dict(torch.load(r'/home/yuantao/code/DGCL/data/CPDB/CPDB_new_data.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = data.to(device)
# Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
# y_all = np.logical_or(data.y, data.y_te)
# mask_all = np.logical_or(data.mask, data.mask_te)
prompt = open("/home/yuantao/code/LLM/txt/prompt/2025加上邻居节点.txt", "r",encoding='gbk').read()
# prompt = open("/home/yuantao/code/LLM/txt/prompt/prompt_new_go.txt", "r",encoding='gbk').read()
cancer_name_txt = pd.read_csv("/home/yuantao/code/LLM/txt/data/cancer.txt", header=None).values

sentance = pd.read_csv(r'/home/yuantao/code/LLM/GO词汇获取/test.csv')
sentance.columns = ['Ensembl_ID','Gene_Symbol','All_Description','Has_Description']

cancer_names = [ 'pan-cancer']
modelname = 'gemma2'

# 移除了全局变量中的MAX_FEATURES定义
CATEGORIES = ['self_statement', 'neighbor_statement', 'together_statement']

for cancer_name in cancer_names:
    cancer_name = cancer_name.upper()
    print(f"Processing {cancer_name}...")
    cancer_name_prompt = cancer_name
    
    # 癌症名称处理保持原样
    for row in cancer_name_txt:
        if row[0].startswith(cancer_name_prompt):
            cancer_name_prompt = row[0].replace("\t", "_")
            break
    
    new_prompt = prompt.replace('Cancer_name', cancer_name_prompt)
    gene_name_list = open(r'/home/yuantao/code/LLM/txt/data/node_names.txt','r',encoding='utf-8').read().split('\n')

    # 输出文件路径保持原样
    output_dir = '/home/yuantao/code/LLM/csv/分类型输出/' + modelname + '/'
    output_file = os.path.join(output_dir, f"{cancer_name}_go_features2.csv")
    os.makedirs(output_dir, exist_ok=True)

    # 修改CSV头生成逻辑
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            header = ['Gene_ID', 'Gene_Symbol'] + CATEGORIES
            f.write(','.join(header) + '\n')
    
    f1 = open(output_file, "a")
    
    # 文件续写逻辑保持原样
    # if os.path.exists(output_file):
    #     with open(output_file, 'r') as f:
    #         lines = f.readlines()
            # if len(lines) == 1:
            #     lenth = 0
            # elif len(lines) < len(gene_name_list):
            #     last_gene = f"{lines[-1].split(',')[0]},{lines[-1].split(',')[1]}"
            #     lenth = gene_name_list.index(last_gene) + 1
            # else:
            #     print('All done')
            #     continue
    
    lenth = 13392
    print('begin from:', gene_name_list[lenth] if lenth < len(gene_name_list) else "END")

    # 基因处理循环
    for idx, gene_entry in enumerate(gene_name_list[lenth:], start=lenth):
        if not gene_entry:
            continue
            
        try:
            gene_id, gene_symbol = gene_entry.split(',', 1)
        except ValueError:
            print(f"跳过无效条目: {gene_entry}")
            continue
        
        retry_count = 0
        success = False
        
        # 节点索引获取保持原样
        try:
            node_idx = data.node_names[:, 1].tolist().index(gene_symbol.strip())
            neighbor_desc = get_neighbor_descriptions(node_idx, data, sentance)
        except ValueError:
            print(f"未找到基因符号: {gene_symbol}")
            neighbor_desc = "No neighboring info"
        
        while retry_count < 3 and not success:
            try:
                # 生成prompt保持原样
                new_prompt1 = new_prompt.replace('Gene_name', gene_symbol)\
                           .replace('go_description', str(sentance['All_Description'][idx]))\
                           .replace('neighbor__info', neighbor_desc)

                # 获取模型响应保持原样
                response = ollama.generate(model=modelname, prompt=new_prompt1)
                content = response['response']
                
                # 处理输出
                feature_dict = process_model_output(content, gene_symbol)
                valid = False
                if feature_dict:
                    # 统计有效字段数
                    valid_count = sum(1 for v in feature_dict.values() if v != 'none')
                    # 有效性规则：至少两个字段有效
                    valid = valid_count >= 2
                
                if valid:
                    # 构建新的CSV行（直接使用类别值）
                    row_data = [
                        gene_id,
                        gene_symbol,
                        f'"{feature_dict["self_statement"]}"',        # 添加引号防止逗号干扰
                        f'"{feature_dict["neighbor_statement"]}"',
                        f'"{feature_dict["together_statement"]}"'
                    ]
                    
                    f1.write(','.join(row_data) + '\n')

                    
                    print(f"成功写入: {gene_symbol}")
                    success = True
                else:
                    print(f"第{retry_count+1}次重试: {gene_symbol}")
                    retry_count += 1
                break
            except Exception as e:
                print(f"处理异常: {str(e)}")
                retry_count += 1
                time.sleep(1)  # 添加重试间隔


            # 定期刷新缓冲区
            if idx % 20 == 0:
                f1.flush()
                print(f"已处理 {idx} 个基因，当前基因: {gene_symbol}")
            
        
    # with open(r'time.txt', 'a') as f:
    #     print(f'model: {modelname}, cancer: {cancer_name}, end time: {datetime.now()}', file=f)
    #     print(f'cost time: {datetime.now() - start}', file=f)
    #     f.close
    f1.close()
    
