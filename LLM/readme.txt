1. go描述获取.ipynb 这个文件是获取放到大语言模型的go语句
2. LLM-go-part.py 获取go的三个方面的词汇，并且保存
	三个方面的词汇会投入biobert，转换之后进入.\data\graph_found.ipynb，变为网络结构，最后的数据为.\data\CPDB\CPDB_merged_k5_edge_index.pt
3. LLM-satment.py 获取context的语句，并且保存，然后投入biobert获得词汇向量，也就是.\data\CPDB\PAN-CANCER_statement_features.pt