import generate_knowledge

# 加载COMET模型
print('generating konwledge model loading...')
comet = generate_knowledge.Comet("./comet-atomic_2020_BART")
comet.model.zero_grad()
print("model loaded")
# relation_set = ["Causes"]
queries1 = []
queries2 = []
rel = "Causes"
train_extended_posts = []  # 存储扩充后的训练文本
test_extended_posts = []  # 存储扩充后的测试文本
i = 1
for tp in train_posts:
    query1 = "{} {} [GEN]".format(tp, rel)
    print(f'{i}已生成')
    queries1.append(query1)
    i = i + 1
results1 = comet.generate(queries1, decode_method="beam", num_generate=5)

# 追加生成结果到train_extended_posts
for i, trp in enumerate(train_posts):  
    train_extended_post = trp 
    for txt in results1[i]:  
        train_extended_post += " " + txt
    train_extended_posts.append(train_extended_post) 

j = 1
for tp in test_posts:
    query2 = "{} {} [GEN]".format(tp, rel)
    print(f'{j}已生成')
    queries2.append(query2)
    j = j + 1
results2 = comet.generate(queries2, decode_method="beam", num_generate=5)

# 追加生成结果到test_extended_posts
for i, tep in enumerate(test_posts):  
    test_extended_post = tep 
    for txt in results2[i]:  
        test_extended_post += " " + txt
    test_extended_posts.append(test_extended_post) 