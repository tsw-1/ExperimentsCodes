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

# System prompt
system_prompt = """You are an AI assistant whose name is EmoLLM.
- EmoLLM is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to to provide psychological counseling for people, especially for counseling, listening, and diagnosis and treatment of depression.
- EmoLLM can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

# 处理每一行数据的函数，使用LLM生成响应并追加
def process_line_with_llm(line):
    prompt = "Please analyze the psychological state of the following post and limit the answer to 20 words or less。post："
    input_text = prompt+line.strip()
    length = 0
    response_text = ""
    # 清理未使用的显存
    torch.cuda.empty_cache()
    for response, _ in model.stream_chat(tokenizer, input_text):
        if response is not None:
            response_text += response[length:]
            length = len(response)
    
    return f"{line.strip()} {response_text.strip()}"

# 处理文件夹中的所有文件
def process_files_in_folder(folder_path, start_index=0):
    files = natsorted(os.listdir(folder_path))
    for idx, fname in enumerate(files):
        if idx < start_index:
            continue
        if fname.endswith('.txt'):
            file_path = os.path.join(folder_path, fname)
            print(f"Processing file: {file_path}")

            processed_lines = []

            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    if line.strip():
                        processed_line = process_line_with_llm(line)
                        processed_lines.append(processed_line)

            with open(file_path, 'w', encoding='utf-8') as file:
                for processed_line in processed_lines:
                    file.write(processed_line + '\n')

# 示例文件夹路径
folder_path = "/E22301339/HAN-BERT/eRisk2017/processed/combined_maxsim16/test"
folder_path2 = "/E22301339/HAN-BERT/eRisk2017/processed/combined_maxsim16/train"
# 处理文件夹中的所有文件
process_files_in_folder(folder_path, start_index=242)
process_files_in_folder(folder_path2)







