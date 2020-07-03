import pandas as pd
import numpy as np

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv("data/twitter_sentiment_analysis.csv", encoding=DATASET_ENCODING, names=DATASET_COLUMNS).loc[:,
     ["target", "text"]]

df.dtypes
df.tail()

pos = df[df["target"] == 4]

idx = np.random.choice(pos.shape[0], 5000)

pos = pos.iloc[idx, :]


i = 0
for i in range(pos.shape[0]):
    pos.iloc[i, 1] = " ".join(filter(lambda x: x[0] != '@', pos.iloc[i, 1].split()))

pos["text"].to_csv("data/twitter.txt", index=False, header=False)