import json
total_user_emotionpost={}
with open('/home/E22301339/depressionDetection/total_user_emotionpost1.json','r')as f:
    total_user_emotionpost.update(json.load(f))
with open('/home/E22301339/depressionDetection/total_user_emotionpost2.json','r')as f:
   total_user_emotionpost.update(json.load(f))
with open('/home/E22301339/depressionDetection/total_user_emotionpost3.json','r')as f:
   total_user_emotionpost.update(json.load(f))
with open('/home/E22301339/depressionDetection/total_user_emotionpost4.json','r')as f:
    total_user_emotionpost.update(json.load(f))
with open('/home/E22301339/depressionDetection/total_user_emotionpost5.json','r')as f:
   total_user_emotionpost.update(json.load(f))
with open('/home/E22301339/depressionDetection/total_user_emotionpost.json','w')as f:
   json.dump(total_user_emotionpost,f)

print("Hello, world!")