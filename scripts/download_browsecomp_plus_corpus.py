from datasets import load_dataset

# 首次运行会下载文档集合。
corpus = load_dataset(
    "Tevatron/browsecomp-plus-corpus",
    split="train",
)

print("文档数量：", len(corpus))
print("字段：", corpus.column_names)

# 保存为本地数据集，之后可以直接从这里加载。
corpus.save_to_disk("./data/browsecomp_plus_corpus")
