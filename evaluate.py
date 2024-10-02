# 对大模型的输出进行eval得到正确率的文件
# 分任务对大模型的输出进行eval
import pandas  as pd 
import glob
import os   
import re
import sys 
sys.path.append("/home/yangchengwu/home2/Hyper_2024/hypergraphqa")
from hypergraph_text_encoder import NODE_ENCODER_DICT
from tqdm import tqdm
def remove_duplicates(dict_list):
    seen = set()
    unique_dicts = []
    for d in dict_list:
        marker = (d['id'], d['text_encoding'])
        if marker not in seen:
            seen.add(marker)
            unique_dicts.append(d)
    
    return unique_dicts
def judge_connected_vertices(gt,output): 
    origin_output = output
    # find gt result from text 
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    index_end = output.find(']')
    if index_begin == -1:
        if "information" in output:
            return False
        else:
            output = ''
            print("wrong")
    else:
        if index_end != -1:
            output = output[index_begin:index_end]
        else:
            output = output[index_begin:]
    matches = re.findall(r'(\d+)', output)
    if len(matches) != 0:
        output = matches 
    else:
        output = re.findall(r"\d+",output)
    output = list(set(output))
    gt = gt.replace('.','')
    gt = gt.split(',')
    try:
        if "No vertices" in gt:
            return len(output) == 0 or "no vertex" in origin_output.lower()
        else:
            return sorted(output,key=int) == sorted(gt,key=int)
    except:
        return False

def judge_vertex_count(gt,output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    if index_begin == -1:
        print("wrong")
    else:
        output = output[index_begin:]
    result = re.findall(r"(\d+)",output)
    # try:
    if len(result) == 0:
        try:
            result = int(re.findall(r"\d+",output)[0])
        except:
            result = 0
    else:
        result = int(result[0])
    gt = int(float(gt))
    return gt == result

def judge_disconnected_vertices(gt,output):
    origin_output = output
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    index_end = output.find(']')
    if index_begin == -1:
        if "information" in output:
            return False
        else:
            output = ''
            print("wrong")
    else:
        if index_end != -1:
            output = output[index_begin:index_end]
        else:
            output = output[index_begin:]
    matches = re.findall(r'(\d+)', output)
    if len(matches) != 0:
        output = matches 
    else:
        output = re.findall(r"\d+",output)
    output = list(set(output))

    gt = gt[:-1].split(',')

    if "No vertices" in gt:
        return len(output) == 0 or "no vertex" in origin_output.lower()
    else:
        return sorted(output,key=int) == sorted(gt,key=int)

def judge_reachability(gt,output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    if index_begin == -1:
        if 'information' in output:
            return False
    else:
        output = output[index_begin:]
    
    output_list = re.findall(r'\w+', output)
    if 'No' in output_list and 'Yes' in output_list:
        return False

    if "No" in gt:
        if gt[:-1] in output:
            return True
        else:
            return False
    else:
        if gt[:-1].lower() in output.lower():
            return True
        else:
            return False


def judge_edge_existence(gt,output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    if index_begin == -1:
        if 'information' in output:
            return False
    else:
        output = output[index_begin:]
    output_list = re.findall(r'\w+', output)
    if 'No' in output_list and 'Yes' in output_list:
        return False
    if "No" in gt:
        if gt[:-1] in output:
            return True
        else:
            return False
    else:
        if gt[:-1].lower() in output.lower():
            return True
        else:
            return False

def judge_vertex_degree(gt,output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    index_end = output.rfind(']')
    if index_begin == -1:
        if 'information' in output:
            return False
        print("wrong")
    else:
        if index_end != -1:
            output = output[index_begin:index_end]
        else:
            output = output[index_begin:]
    result = re.findall(r"(\d+)",output)
    if len(result) == 0:
        result = 0
    else:
        result = int(result[0])
    gt = int(float(gt))
    return gt == result

def judge_shortest_path(gt,output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    if index_begin == -1:
        if 'information' in output:
            return False
        print("wrong")
    else:
        output = output[index_begin:]
    if "There is no path from " in gt:
        return "no path" in output.lower()
    try:
        matches = re.findall(r'(\d+)', output)
        matches = int(matches[0])
    except:
        matches = 0
    gt = int(float(gt))
    return gt == matches

def judge_edge_count(gt,output):
    gt =  int(float(gt))
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    # index_end = output.rfind(']')
    if index_begin == -1:
        print("wrong")
    else:
        output = output[index_begin:]
    result = re.findall(r"(\d+)",output)
    try:
        if len(result) == 0:
            result = int(re.findall(r"\d+",output)[0])
        else:
            result = int(result[0])
    except:
        result = 0
    return gt == result

def judge_set_connection(gt,output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    index_end = output.find(']')
    if index_begin == -1:
        if 'information' in output:
            return False
        # print("wrong")
    else:
        if index_end == -1:
            output = output[index_begin:]
        else:
            output = output[index_begin:index_end]
    output_list = re.findall(r'\w+', output)
    if 'No' in output_list and 'Yes' in output_list:
        return False
    if "No" in gt:
        if gt[:-1] in output:
            return True
        else:
            return False
    else:
        if gt[:-1].lower() in output.lower():
            return True
        else:
            return False 

judge_set_existence = judge_set_connection

def parse_prediction_hypergraph(output):
    begin_temp = ['Ans','A','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break 
    # index_end = output.rfind(']')
    if index_begin == -1:
        return output
    else:
        output = output[index_begin:]
    return output

def convert_text_to_int(output,graph_text):
    for i,val in enumerate(NODE_ENCODER_DICT[graph_text].values()):
        output = output.replace(val,str(i))
    return output

def judge_shape_prediction(gt,output):
    gt =  int(float(gt[:-1]))
    begin_temp = ['Ans','A:','[','the answer is:']
    for i,temp in enumerate(begin_temp):
        index_begin = output.find(temp)
        if index_begin != -1:
            break
    # index_end = output.rfind(']')
    if index_begin == -1:
        print("wrong")
        if 'Prompt tokens' in output or 'max input' in output:
            return False
    else:
        output = output[index_begin:]
    result = re.findall(r"(\d+)",output)
    try:
        result = int(result[0])
    except:
        result = -1
    return gt == result

EVAL_SOLOVER = {
    'hyperedge_count':judge_edge_count,
    'vertex_count':judge_vertex_count,
    'vertex_degree':judge_vertex_degree,
    'vertex_connection':judge_edge_existence,
    'reachability':judge_reachability,
    'shortest_path':judge_shortest_path,
    'connected_vertices':judge_connected_vertices,
    'disconnected_vertices':judge_disconnected_vertices,
    'hyperedge_degree':judge_vertex_degree,
    'vertexset_connection':judge_set_connection,
    'vertexset_hyperedge':judge_set_existence,
    'hyperedge_hyperedge':judge_edge_existence,
    'shared_vertices':judge_connected_vertices,
    'isomorphism':judge_edge_existence,
    "shape_prediction":judge_shape_prediction,
}

from multiprocessing import Pool
from multiprocessing import Manager

def eval_muti_process(qa):
    gt = str(qa['answer'])
    graph_text = qa['text_encoding']
    output = qa['response'] if 'response' in qa.keys() else qa['output']
    if graph_text not in text_encodding:
        text_encodding[graph_text] = {'correct':0,'total':0,'TP':0,'TN':0,'FP':0,'FN':0}
    tmp = text_encodding[graph_text]
    for i,val in enumerate(NODE_ENCODER_DICT[graph_text].values()):
        output = output.replace(val,str(i))
        gt = gt.replace(val,str(i))
    if 'reachability' in name or 'isomorphism' in name:
        solver = name.split('_')[0] 
    else:
        solver = name.split('_')[0] + '_' +name.split('_')[1]
    solver = EVAL_SOLOVER[solver]
    output = output.replace('[Yes, No,]','')
    if solver(gt=gt,output=output):
        qa['result'] =  True
        tmp['correct'] += 1 
    else:
        qa['result'] = False
    tmp['total'] += 1 
    text_encodding[graph_text] = tmp


if __name__ == '__main__':
    ret = {}
    path_list = glob.glob("")
    for j,path in enumerate(path_list):
        name = os.path.basename(path).split('.')[0]
        df = pd.read_csv(path)
        list_of_dicts = df.to_dict(orient='records')
        list_of_dicts = [i for i in list_of_dicts if 'Prompt tokens too long' not in str(i['response']) and 'context length error' not in str(i['response']) and 'nan' != str(i['response'])]
        text_encodding = Manager().dict()
        with Pool(processes=1) as pool:
            list(tqdm(pool.imap(eval_muti_process, list_of_dicts,chunksize=1), total=len(list_of_dicts), desc=f'File:{len(list_of_dicts)}{name}:{j}/{len(path_list)}'))
        text_encodding = dict(text_encodding)
        for key,value in text_encodding.items():
            if key not in ret:
                ret[key] = {}
            tmp = ret[key]
            tmp[name] = value['correct'] / (value['total']+1e-8)
    
    desired_order = ['zero_shot', 'zero_cot', 'few_shot', 'cot','cot_bag']
    desired_order_2 = ['hyperedge_count',
                        'vertex_count',
                        'vertex_degree',
                        'vertex_connection',
                        'reachability',
                        'shortest_path',
                        'connected_vertices',
                        'disconnected_vertices',
                        'hyperedge_degree',
                        'vertexset_connection',
                        'vertexset_hyperedge',
                        'hyperedge_hyperedge',
                        'shared_vertices',
                        'isomorphism',]
    def extract_substring(key, substrings,substrings2):
        index = len(substrings)
        max_len = 0
        for substring in substrings:
            if substring in key:
                if len(substring) > max_len:
                    index = substrings.index(substring)
                    max_len = len(substring)
        
        index2 = len(substrings2)
        max_len2 = 0
        for substring in substrings2:
            if substring in key:
                if len(substring) > max_len:
                    index2 = substrings2.index(substring)
                    max_len2 = len(substring)
        return index*len(substrings2) + index2
    from collections import OrderedDict
    ret_order = {}
    for key in ret.keys():
        value = ret[key]
        sorted_keys = sorted(value.keys(), key=lambda k: extract_substring(k, desired_order,desired_order_2))
        sorted_dict = OrderedDict()
        for k in sorted_keys:
            sorted_dict[k] = value[k]
        ret_order[key] = sorted_dict
    for key,value  in ret_order.items():
        for k,v in value.items():
            print(f'encoding:{key},name:{k},acc:{round(v,3)}')
        seq_mark = -1 
        for k,v in value.items():
            tmp = extract_substring(k,desired_order,[''])
            if seq_mark < tmp:
                seq_mark = tmp                
            print(f'{round(v,3)}',end=' ')
        print(f'\n******************************************\n')

    desired_order = ['zero_shot', 'zero_cot', 'few_shot', 'cot','cot_bag']
        