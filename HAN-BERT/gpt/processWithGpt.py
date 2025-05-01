import os
import json
import requests
from natsort import natsorted
from tqdm import tqdm

# 定义获取 GPT 响应的函数
def get_chat_gpt_response(prompt):
    url = "https://api.gptgod.online/v1/chat/completions"
    headers = {
        "Authorization": "sk-MDpQ3NHw2IfRkheX1zBeycCEImpS0jAzyQXLS63u9DPsprON",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "Hello, you are now an experienced psychologist and psychiatrist, especially good at assisting in the diagnosis of depression. Please evaluate the depression risk of the user's post I will provide. Answer format: {\"Depression\":\"Yes\",\"Symptoms\":[\"Sadness\",\"Anxiety disorder diagnosis\"],\"Score\":8}, {\"Depression\":\"No\",\"Symptoms\":[],\"Score\":0}."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return "error"+str(e)

def process_single_json_file(filepath, output_filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data_gpt = []
        for post in tqdm(data, desc=f"Processing {os.path.basename(filepath)}", unit="posts"):
            for post_id, post_content in post.items():
                if post_id not in ["Depression", "Symptoms", "Scores"]:
                    # 执行 GPT 分析
                    try:
                        gpt_response_content = get_chat_gpt_response(post_content)
                        gpt_response = json.loads(gpt_response_content)
                        post_gpt = {
                            post_id: post_content,
                            "Depression": gpt_response.get("Depression", ""),
                            "Symptoms": gpt_response.get("Symptoms", ""),
                            "Scores": gpt_response.get("Score", "")
                        }
                    except json.JSONDecodeError:
                        post_gpt = {
                            post_id: post_content,
                            "Depression": "",
                            "Symptoms": "",
                            "Scores": "",
                            "exception": gpt_response_content
                        }
                    data_gpt.append(post_gpt)

    # 将处理后的结果写入一个新的 JSON 文件
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data_gpt, f, ensure_ascii=False, indent=4)


# 处理目录中的所有 JSON 文件
def process_json_files_in_directory(directory):
    for subdir in ['test', 'train']:
        subdir_path = os.path.join(directory, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        for file in natsorted(os.listdir(subdir_path)):
            filepath = os.path.join(subdir_path, file)
            if filepath.endswith(".json"):
                output_filepath = os.path.join(subdir_path, f"gpt_{file}")
                process_single_json_file(filepath, output_filepath)

process_json_files_in_directory('/home/E22301339/scale_early_depress_detect-main/eRisk2017/processed/combined_maxsim32_json')
