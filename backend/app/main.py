# # Entry point for your backend server.
# from huggingface_hub import login
# login()

# from datasets import load_dataset

# ds = load_dataset("community-datasets/proto_qa", "proto_qa")
import pandas as pd

splits = {'train': 'proto_qa/train-00000-of-00001.parquet', 'validation': 'proto_qa/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/community-datasets/proto_qa/" + splits["train"])