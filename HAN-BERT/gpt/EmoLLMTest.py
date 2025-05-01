from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_dir = '/home/E22301339/EmoLLM_aiwei'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

input_text = "介绍一下自己，心理大模型?"
# inputs = tokenizer(input_text, return_tensors="pt")
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# outputs = model.generate(input_ids,max_length=64)
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_text) # 生成文本回答

# 得到语义嵌入
# 使用模型生成文本的隐藏状态
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
# 获取最后一层的隐藏状态作为文本嵌入
text_embedding = hidden_states[-1].squeeze(0)  # 去除批处理维度
print("ending................................................................")


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
