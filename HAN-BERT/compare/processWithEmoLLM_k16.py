import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from testNasorted import natsorted

# 初始化模型和设备
model_dir = "/E22301339/EmoLLM_aiwei"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = model.eval()

# 设置环境变量以优化显存分配
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# System prompt
system_prompt = """You are an AI assistant whose name is EmoLLM.
- EmoLLM is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to to provide psychological counseling for people, especially for counseling, listening, and diagnosis and treatment of depression.
- EmoLLM can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
messages = [(system_prompt, 'system')]

def evaluate_depression_score(line):
    prompt = """你现在是一名经验丰富的心理学家和精神病学家，尤其擅长抑郁症的辅助诊断。请帮我基于DSM-5标准分析下面的帖子是否出现了抑郁症状。不需要任何额外输出，回答格式为一个json字符串。给出两个输出示例：{"抑郁症":"是","抑郁症状":["沮丧","自杀倾向"],"得分":7}、{"抑郁症":"否","抑郁症状":[],"得分":0}。 post："""
    max_length = 512  # 设置最大输入长度
    input_text = (prompt + line.strip())[:max_length]
    length = 0
    response_text = ""

    torch.cuda.empty_cache()

    for response, _ in model.stream_chat(tokenizer, input_text,messages):
        if response is not None:
            response_text += response[length:]
            length = len(response)

    torch.cuda.empty_cache()

    # 提取分数和症状信息
    try:
        response_data = eval(response_text.strip())
        if isinstance(response_data, list) and len(response_data) == 3:
            score = int(response_data[-1])
            symptoms = response_data[1]
        else:
            score = 0
            symptoms = "无抑郁症状"
    except (ValueError, SyntaxError):
        score = 0  # 处理无法解析的情况
        symptoms = "无抑郁症状"

    return score, symptoms

def process_files_in_folder(folder_path, start_index=0):
    posts_with_scores = []

    files = natsorted(os.listdir(folder_path))
    for idx, fname in enumerate(files):
        if idx < start_index:
            continue
        if fname.endswith('.txt'):
            file_path = os.path.join(folder_path, fname)
            print(f"Processing file: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        score, symptoms = evaluate_depression_score(line)
                        posts_with_scores.append((line.strip(), score, symptoms))
    
    # 按分数从高到低排序，并筛选出排名前16的帖子
    top_posts = sorted(posts_with_scores, key=lambda x: x[1], reverse=True)[:16]
    
    # 保存结果
    output_file_path = os.path.join(folder_path, "top_16_high_risk_posts.txt")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for post, score, symptoms in top_posts:
            output_file.write(f"Score: {score}\nSymptoms: {symptoms}\nPost: {post}\n\n")

folder_path = "/E22301339/HAN-BERT/eRisk2017/processed/combined_maxsim32/test"
process_files_in_folder(folder_path, start_index=0)  # 从第113个文件开始处理
