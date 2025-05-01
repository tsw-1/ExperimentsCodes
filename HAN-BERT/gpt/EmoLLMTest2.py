import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
 
 
model_dir = "/home/E22301339/EmoLLM_aiwei"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
model = model.eval()
 
system_prompt = """You are an AI assistant whose name is EmoLLM.
- EmoLLM is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to to provide psychological counseling for people, especially for counseling, listening, and diagnosis and treatment of depression.
- EmoLLM can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
 
messages = [(system_prompt, '')]
 
print("=============Welcome to EmoLLM chatbot, type 'exit' to exit.=============")
 
while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break
 
    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)