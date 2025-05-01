import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
 
model_dir = "/E22301339/EmoLLM_aiwei"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model = model.eval()

 
system_prompt = """You are an AI assistant whose name is EmoLLM.
- EmoLLM is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to to provide psychological counseling for people, especially for counseling, listening, and diagnosis and treatment of depression.
- EmoLLM can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
messages = [(system_prompt, '')]
 
# 打印欢迎信息
print("============= Welcome to EmoLLM chatbot =============")
print("Type 'exit' to exit the chat.")
print("------------------------------------------------------")
 
while True:
    # 获取用户输入
    input_text = input("\nUser: ").strip()
    if input_text.lower() == "exit":
        print("Goodbye! Have a nice day!")
        break

    # 初始化响应长度
    length = 0
    print("EmoLLM:", end=" ")

    # 生成并打印响应
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

    print("\n------------------------------------------------------")