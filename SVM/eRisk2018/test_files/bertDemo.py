# %%
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-distilroberta-base-v1',device='cuda')
sentences=['''I'm pretty sure the definition is "It's not slander if it's true" - if it wasn't true that they were doing this, they could always do like other developers have done and come to the subreddit to make some explanation, like the games that had just recently gone out of early access entirely coincidentally - if they've got no excuse and they're just trying to cover it up by falsely flagging the video then you'd assume they are doing something underhanded here''','''To be fair, it is pretty subjective, depends on what kinds of games OP likes''','''So far - 827.  Didn't get to play at all before this since it wouldn't let me in any games, so seems not too bad.''','''Happened to me once earlier - assuming you're playing on the client, close the whole thing and try reopening it, that worked for me.''','''Honestly I'm not even sure why companies would want to do this.  People aren't stupid, and if customers feel like they're being ripped off or cheated then they're less likely to want to support a company in the future.  And if they aren't doing something shady, there are ways to correct misinformation that *don't* feel like a coverup.  The "coincidental price increase because leaving early access" games have handled this pretty well.  The ones that haven't, though - illegal or not, it just seems dumb, from a customer respect/trust standpoint''','''If you go to account details and then hit store transactions, it'll show a list of your recent purchases.  The one problem with that is if you buy multiple games at a time, it shows the total purchase price instead of the cost for each individual game.  But I'm pretty sure you can see the cost for each game by checking the confirmation email steam sends when you buy something.''','''Man, I can at least hope that the non-stupid people outweigh the stupid ones by some amount, even though that's probably optimistic and I'm probably wrong.''','''You're not going crazy, I heard it.  Just once, and I don't remember which level it was.  I kind of like it over the regular boss music''','''My name's uncommon enough in the US that I'm pretty much 100%% sure it won't be in''','''I want to wear a pip-boy like the irredeemable nerd I am''']
#%%
questionaire_single = [ #   BDI-II 抑郁症测量表定义的症状----抑郁症模板中的第二组
    "I feel sad.",
    "I am discouraged about my future.",
    "I always fail.",
    "I don't get pleasure from things.",
    "I feel quite guilty.",
    "I expected to be punished.",
    "I am disappointed in myself.",
    "I always criticize myself for my faults.",
    "I have thoughts of killing myself.",
    "I always cry.",
    "I am hard to stay still.",
    "It's hard to get interested in things.",
    "I have trouble making decisions.",
    "I feel worthless.",
    "I don't have energy to do things.",
    "I have changes in my sleeping pattern.",
    "I am always irritable.",
    "I have changes in my appetite.",
    "I feel hard to concentrate on things.",
    "I am too tired to do things.",
    "I have lost my interest in sex.",
    
]

#%%
embeddings1 = model.encode(sentences, convert_to_tensor=True)
embeddings2=model.encode(questionaire_single,convert_to_tensor=True)
# print(embeddings1,embeddings1.size)

#%%
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
print(cosine_scores)
# %%
# str="It's a good day,and I hava a good time"
# str_embedding=model.encode(str,convert_to_tensor=True)
# scores=util.pytorch_cos_sim(str_embedding,embeddings2)
# %%
