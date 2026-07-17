# 标准库
import os
import joblib

# 科学计算库
import numpy as np

# 检索相关
import faiss
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

# 注释类
from langchain_core.documents import Document

# 项目内部工具
from utils.path import PROJECT_PATH
from scripts.rag.process_paper import process_allPapers
from scripts.rag.process_paper import embed_model


"""拼接documents和embeddings，构建faiss"""
process_allPapers()   # 拼接前先看有没有未处理的论文，如果有就先处理了先

docs = []
embeds = []

# 遍历所有论文，拼接documents和embeddings
folder_path = os.path.join(PROJECT_PATH, "data/processed_papers")
paper_name_list = os.listdir(folder_path)
if ".gitkeep" in paper_name_list:
    paper_name_list.remove(".gitkeep")

for paper_name in paper_name_list:
    paper_folder = os.path.join(folder_path, paper_name)

    # 先检查有没有docs和embeds，然后再读取
    docs_path = os.path.join(paper_folder, "docs.pkl")
    embeds_path = os.path.join(paper_folder, "embeds.pkl")
    if os.path.exists(docs_path) is True and os.path.exists(embeds_path) is True:
        paper_docs = joblib.load(docs_path)
        paper_embeds = joblib.load(embeds_path)

        # 拼接
        docs.extend(paper_docs)
        embeds.append(paper_embeds)

# 处理embeds
embeds = np.vstack(embeds)

# 构建faiss
dimension = embeds.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeds)


"""读取embedding模型"""
"""读取rerank模型"""
rerank_model_dir = snapshot_download(
    model_id="AI-ModelScope/bge-reranker-v2-m3",
    cache_dir=os.path.join(PROJECT_PATH, "models"),
)
rerank_model = CrossEncoder(rerank_model_dir, device="cuda")


def retrieval_docs_by_vector(query: str, top_k: int) -> list[Document]:
    """
    根据余弦相关性获取top_k个documents，
    由于该文件已经准备好了docs、embeds和faiss，所以只需传入用户的问题与top_k
    :param query: 用户的问题
    :param top_k: 获取最相关documents的数量
    :return: 返回最相关的docs
    """
    # 对question做embedding
    question_embed = embed_model.encode([query])

    # 检索并获取最相关的k个embedding的indices
    _, indices = index.search(question_embed, top_k)   # 获取top_k的索引
    top_docs = [docs[idx] for idx in indices[0]]   # 根据索引提取相关docs

    return top_docs


def retrieval_docs_by_BM25(query: str, top_k: int) -> list[Document]:
    """
    根据BM25获取top_k个documents，
    由于该文件已经准备好了docs、embeds和faiss，所以只需传入用户的问题与top_k
    :param query: 用户的问题
    :param top_k: 获取最相关documents的数量
    :return: 返回最相关的docs
    """
    # 使用bm25进行检索
    retriever = BM25Retriever.from_documents(documents=docs)
    retriever.k = top_k
    top_docs = retriever.invoke(query)

    return top_docs


def rerank_docs(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """
    使用rerank的方式，从docs里找出top_k个与query最相关的document
    :param query: 用户的问题
    :param docs: 需要rerank的docs
    :param top_k: rerank的top_k
    :return: 返回rerank后的docs
    """
    # rerank
    rerank_input = [[query, doc.page_content] for doc in docs]   # 构建rerank输入
    rerank_score = rerank_model.predict(rerank_input)

    # 根据分数排序，获取top_k的docs
    sorted_idx_list = np.argsort(rerank_score)[::-1]
    top_idx_list = sorted_idx_list[:top_k]
    top_docs = [docs[idx] for idx in top_idx_list]

    return top_docs


def retrieval_docs(query: str, vector_top_k: int = 10, bm25_top_k: int = 10, rerank_top_k: int = 3) -> list[Document]:
    """
    整合vector、bm25、rerank的召回结果，获取最相关的docs
    :param query: 用户的问题
    :param vector_top_k: 根据余弦相似度retrieval的top_k
    :param bm25_top_k: 根据bm25去retrieval的top_k
    :param rerank_top_k: rerank的top_k
    :return: 最终retrieval的docs
    """
    # 获取根据vector和bm25检索的docs
    vector_docs = retrieval_docs_by_vector(query, vector_top_k)
    bm25_docs = retrieval_docs_by_BM25(query, bm25_top_k)
    docs = vector_docs + bm25_docs

    # rerank
    top_docs = rerank_docs(query, docs, rerank_top_k)

    return top_docs


if __name__ == "__main__":
    """
    这个文件负责构建与检索相关的函数，供之后的RAG的tool使用
    """

    # test
    print(len(docs), embeds.shape)
    query = "臭氧的生成机制有哪些"

    vector_docs = retrieval_docs_by_vector(query, 10)
    bm25_docs = retrieval_docs_by_BM25(query, 10)
    reranked_docs = rerank_docs(query, vector_docs + bm25_docs, 3)
    print(len(vector_docs), len(bm25_docs), len(reranked_docs))
