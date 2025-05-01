from transformers import AutoTokenizer, AutoModel

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained("/home/E22301339/scale_early_depress_detect-main/mental-bert")
model = AutoModel.from_pretrained("/home/E22301339/scale_early_depress_detect-main/mental-bert")

# 输入文本
text = "what a good day!"

# 使用分词器对文本进行编码
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# 将编码的文本输入模型进行推理
outputs = model(**inputs)

# 提取模型的输出，通常是隐藏状态
# 例如，提取最后一层的隐藏状态
last_hidden_state = outputs.last_hidden_state

# 打印编码的文本表示
print(last_hidden_state)
