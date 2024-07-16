import pandas as pd

splits = {'train': 'proto_qa/train-00000-of-00001.parquet', 'validation': 'proto_qa/validation-00000-of-00001.parquet'}

train = pd.read_parquet("hf://datasets/community-datasets/proto_qa/" + splits["train"])
val = pd.read_parquet("hf://datasets/community-datasets/proto_qa/" + splits["validation"])

training_set = train.drop(['question' ,'answer-clusters','totalcount', 'id', 'source'], axis=1)
validation_set = val.drop(['question' ,'answer-clusters','totalcount', 'id', 'source'], axis=1)

training_set.rename(columns={'normalized-question': "question", 'answerstrings': "answers"}, inplace=True)
validation_set.rename(columns={'normalized-question': "question", 'answerstrings': "answers"}, inplace=True)

combined = pd.concat([training_set, validation_set], ignore_index=True)

# print(combined.head(1))

if __name__=="__main__":
  
  combined.to_pickle('dataset.pkl')