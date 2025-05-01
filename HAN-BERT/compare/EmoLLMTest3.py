import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_dir = "/E22301339/EmoLLM_aiwei"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

# 将模型移至设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = model.eval()

# 定义系统提示和初始消息
system_prompt = """You are an AI assistant whose name is EmoLLM.
- EmoLLM is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to to provide psychological counseling for people, especially for counseling, listening, and diagnosis and treatment of depression.
- EmoLLM can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
messages = [(system_prompt, 'system')]

# 打印欢迎信息
print("============= Welcome to EmoLLM chatbot =============")
print("Type 'exit' to exit the chat.")
print("------------------------------------------------------")

while True:
    # 获取用户输入
    user_input = input("\nUser: ").strip()
    if user_input.lower() == "exit":
        print("Goodbye! Have a nice day!")
        break

    # 添加用户消息到上下文
    messages.append((user_input, 'user'))

    # 初始化响应长度
    length = 0
    print("EmoLLM:", end=" ")

    # 准备用于生成响应的输入
    chat_history = system_prompt + '\n' + '\n'.join([f"{role}: {text}" for text, role in messages])

    # 生成并打印响应
    response_text = ""
    for response, _ in model.stream_chat(tokenizer, chat_history, max_new_tokens=1000, stop_token=None):
        if response is not None:
            print(response[length:], flush=True, end="")
            response_text += response[length:]
            length = len(response)

    # 添加模型回复到上下文
    messages.append((response_text, 'assistant'))

    print("\n------------------------------------------------------")
