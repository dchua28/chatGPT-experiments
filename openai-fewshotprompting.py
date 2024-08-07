import os
import numpy as np
import pandas as pd
from openai import OpenAI

client = OpenAI( api_key=os.environ.get("OPENAI_API_KEY"),)

data_file = "imdb_10K_sentiments_reviews.csv"
df = pd.read_csv(data_file, encoding='utf-8')
df['label'] = df['label'].replace({0: 'Negative', 1: 'Positive'})
df.head()

system_message = """
You are a binary classifier for sentiment analysis.
Given a text, based on its sentiment, you classify it into 
one of two caterories: positive or negative.
You can use the following texts as examples:
Text: "I am pleasantly surprised by this fountain pen, the weight and feel in the hand is substantial and the ink distribution is smooth and solid, no skips."
Positive
Text: "What was delivered was a burn piece of goat cheese and 9 small cubes of squash."
Negative
Text: "Easy to assemble. Sturdy and looks good. Was perfect solution for story my many plants."
Positive
Text: "The printed code style (grey with horrible white lines) is really disturbing for the eye."
Negative
ONLY return the sentiment as output (without punctuation).
Text:
"""

df_sampling = df.sample(n=10, random_state=12)

def classify_review(review):
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": system_message},
      {"role": "user", "content": review}
    ]
  )
  return completion.choices[0].message.content

df_sampling['predicted'] = df_sampling['review'].apply(classify_review)
print(df_sampling)