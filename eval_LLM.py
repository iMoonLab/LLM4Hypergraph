import requests
import json
from http import HTTPStatus
import time
import pandas as pd 
import glob
from multiprocessing import Pool
from multiprocessing import Manager
import os 
from tqdm import tqdm
from openai import AzureOpenAI
import dashscope
from functools import partial


def call_with_messages_gpt(QA,model="gpt35"):
    question = QA['question']
    client = AzureOpenAI(
    azure_endpoint = "https://llm4hypergraph.openai.azure.com/", 
    api_key="",  
    api_version="2024-02-01"
    )
    while True:
        try:
            response = client.chat.completions.create(
                model=model, 
                messages=[
                    {
                        "role": "user", 
                        "content": question
                    },
                ]
            )
            QA.update({'response':response.choices[0].message.content,'result':True})
            save_list.append(QA)  
            break
        except:
            time.sleep(60)


def call_with_messages_llama(QA,model="meta-llama/llama-3-8b-instruct"):
    question = QA['question']
    while True:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer sk-",
                },
                data=json.dumps({
                    "model": model, 
                    "messages": [
                    { "role": "user", "content": question}
                    ]
                    
                })
            )
            if 'error_msg' not in json.loads(response.text) and response.status_code == HTTPStatus.OK:
                response_content = json.loads(response.text)['choices'][0]['message']['content']
                QA.update({'response':response_content,'result':True})
                save_list.append(QA)   
                break
            else:
                time.sleep(10)
        except:
            time.sleep(10)


def call_with_messages_qwen(QA,model="qwen-long"):
    # with semaphore:
    question = QA['question']
    messages = [
        {'role': 'user', 'content': question}]
    dashscope.api_key  = ''
    while True:
        response = dashscope.Generation.call(
            model= model,
            messages=messages,
            result_format='message',  
        )
        if response.status_code == HTTPStatus.OK:
            while time.time() - begin < 1:
                time.sleep(0.1)
            break
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))
            time.sleep(20)
    try:
        response_content = response['output']['choices'][0]['message']['content']
    except: 
        response_content = ''
    QA.update({'response':response_content,'result':True})
    save_list.append(QA)    


def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = ""
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


def call_with_messages_baidu(QA,model="ernie-lite-8k"):
    question = QA['question']
    answer = QA['answer']
    while True:
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k?access_token=" + get_access_token()
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })
        headers = {
        'Content-Type': 'application/json'
        }
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if 'error_msg' not in json.loads(response.text) and response.status_code == HTTPStatus.OK:
                # print(response)
                while time.time() - begin < 1:
                    time.sleep(0.1)
                response_content = json.loads(response.text)['result']
                break
            else:
                print('error message: %s' % (
                    json.loads(response.text)['error_msg']
                ))
                time.sleep(10)
        except:
            time.sleep(10)
    QA.update({'response':response_content,'result':str(str(answer) in response_content)})
    save_list.append(QA)    

def remove_duplicates(dict_list):
    seen = set()
    unique_dicts = []
    for d in dict_list:
        marker = (d['id'], d['text_encoding'])
        if marker not in seen:
            seen.add(marker)
            unique_dicts.append(d)
    return unique_dicts


LLMS = {
    "meta-llama/llama-3-8b-instruct":call_with_messages_llama,
    "gpt35": call_with_messages_gpt,
    "gpt4": call_with_messages_gpt,
    "qwen-long": call_with_messages_qwen,
    "ernie-lite-8k_low":call_with_messages_baidu,   
}

model = "meta-llama/llama-3-8b-instruct"
result_dir = model
if __name__ == '__main__':
    file_path_list = sorted(glob.glob("."))
        
    for j,file_path in enumerate(file_path_list):
        root = os.path.dirname(os.path.dirname(file_path))
        name = os.path.basename(file_path).split('.')[0]
        save_list = Manager().list()
        loaded_object = pd.read_csv(file_path)
        loaded_object = loaded_object.loc[:, ~loaded_object.columns.str.contains('^Unnamed')].to_dict(orient='records')
        if len(loaded_object) == 0:
            continue
        processes = []
        num_processes = 4
        begin = time.time()
        func = partial(LLMS[model], model=model)
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(func, loaded_object,chunksize=1), total=len(loaded_object), desc=f'File:{name}:{j}/{len(file_path_list)}'))
        for p in processes:
            p.join()
        end = time.time()
        print(f"times:{end-begin}s")
        save_list = list(save_list)
        df = pd.DataFrame(save_list)
        result_dir_pt =  result_dir 
        try:
            df_ori = pd.read_csv(os.path.join(f'{root}/{result_dir_pt}',name+'.csv'))
            df_ori = df_ori.loc[:, ~df_ori.columns.str.contains('^Unnamed')]
            df = pd.concat([df,df_ori],ignore_index=True)
        except:
            pass
        df = pd.DataFrame(remove_duplicates(df.to_dict(orient='records')))
        os.makedirs(f'{root}/{result_dir_pt}',exist_ok=True)
        df.to_csv(os.path.join(f'{root}/{result_dir_pt}',name+'.csv'),
                columns=save_list[0].keys(),
                header=save_list[0].keys(),
                )
