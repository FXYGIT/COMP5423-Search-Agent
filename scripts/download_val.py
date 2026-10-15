from datasets import load_dataset

ds = load_dataset("HongruCai/COMP5423-BrowseComp-Plus-Val")
ds.save_to_disk("./dataset-val")
