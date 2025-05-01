import json
with open('/home/E22301339/depressionDetection/total_user_wordpost_chunks.json','r')as f:
    posts=json.load(f)
# 将dict的key按照顺序保存到一个列表中
keys_list = list(posts.keys())

# 将前四份的大小设置为100，最后一份的大小设置为86
part_size = 100
last_part_size = 86

# 用列表切片将dict分成五份
part1 = {k: posts[k] for k in keys_list[:part_size]}
part2 = {k: posts[k] for k in keys_list[part_size:2 * part_size]}
part3 = {k: posts[k] for k in keys_list[2 * part_size:3 * part_size]}
part4 = {k: posts[k] for k in keys_list[3 * part_size:4 * part_size]}
part5 = {k: posts[k] for k in keys_list[-last_part_size:]}

print("================================================")
with open('/home/E22301339/depressionDetection/total_user_wordpost_chunks_part1.json','w')as f:
    json.dump(part1, f)
with open('/home/E22301339/depressionDetection/total_user_wordpost_chunks_part2.json','w')as f:
    json.dump(part2, f)
with open('/home/E22301339/depressionDetection/total_user_wordpost_chunks_part3.json','w')as f:
    json.dump(part3, f)
with open('/home/E22301339/depressionDetection/total_user_wordpost_chunks_part4.json','w')as f:
    json.dump(part4, f)
with open('/home/E22301339/depressionDetection/total_user_wordpost_chunks_part5.json','w')as f:
    json.dump(part5, f)