import requests


def get_chat_gpt_response(prompt):
    url = "https://api.gptgod.online/v1/chat/completions"
    headers = {
        "Authorization": "sk-bBLnlH3a9sBGdOA9QaboOTd0y78jiXeBdacYVMs8TdX8WFhr",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4-gizmo-g-ul7aqZCz2",
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()


input_text2=""" 分析一下下面发帖人的心理状态，以及可能存在的心理问题（抑郁症），要求用中文回答，20字以内。帖子内容：At age 30, I was carjacked, robbed and raped. My family still refuses to talk about it.  I've moved on the best I can, and time has helped.  A few years later, there was a serial rapist that matched my perps MO and description.  After a few drinks, I went looking for him.  No property was damaged and no one was hurt, but I paid my debt to society.  Believe me, I wanted to be punished. Family was predictably not supportive (which I understand, I broke the law.). But it was a difficult time for me to say the least.  My husband decided to throw a surprise party for me this Friday.  None of my blood relatives are attending.  We don't live that far away, but yet my mother told my husband via text that she felt "I was pulling away and they couldn't come because my dad is tired."  They also refused to watch my brother's kids, so my brother and sister in law are not coming.  Never mind the fact that I am constantly running up there for all their parties and created an education trust so that my nephew will have options when he turns 18.  I gave him my old car when they got "accidentally" pregnant with their second child.  This is long, but I'm at the end of my rope. I've been actively suicidal before, and I feel it creeping up on me again.  I've been through 10 therapists through no fault of my own, they're always changing practices.  My p-doc diagnosis of me is major depression and anxiety.  I disagree.  I believe I may have bi-polar, possibly borderline, and for sure PTSD.  All of this is just to say, what's the point?  No kids, job I hate, family who is only interested in what they can get out of me and this pseudo-marriage.    
"""
input_text=""" Post: It sent me into a full-blown crisis by breaking open the “ no one believes me and I'm worthless"wound.I had to up my level of care, now I'm in lOP 5 days a week, 3+ hours a day. It's given me just enough relief to return to weekly EMDR. I had my first session in this round, last week. It was the most physically symptomatic I've ever been in-session.
Consider this post on social media to answer the question: Determine how depressed the person is. Return one item from: minimum, mild, moderate, severe. Provide reasons step by step. """
response = get_chat_gpt_response(input_text)
print(response['choices'][0]['message']['content'])

