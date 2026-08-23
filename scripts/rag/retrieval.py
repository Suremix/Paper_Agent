# 标准库
import os
import joblib
from dotenv import load_dotenv

# 科学计算库
import numpy as np

# 检索相关
import faiss
import torch
from modelscope import snapshot_download
from sentence_transformers import CrossEncoder
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

# 注释类
from langchain_core.documents import Document

# 内部
from scripts.paper.process_papers import process_papers
from scripts.paper.encode_papers import encode_model
from scripts.rag.prompt import translate_prompt
from utils.path import PROJECT_PATH

load_dotenv()   # 读取api

# 读取rerank模型
model_id = "AI-ModelScope/bge-reranker-v2-m3"
rerank_model_dir = snapshot_download(
    model_id=model_id,
    cache_dir=os.path.join(PROJECT_PATH, "models")
)
device = "cuda" if torch.cuda.is_available() else "cpu"   # 检查是否可以使用cuda
rerank_model = CrossEncoder(rerank_model_dir, device=device)


class BaseRetriever:
    """负责整合较为基础的召回算法，以及数据读取"""
    def __init__(self):
        self.docs = None
        self.embeds = None
        self.index = None

        self.load_data()   # 读取论文库数据

    def load_data(self):
        """负责读取docs和embeds"""
        process_papers()  # 如果有还未被处理的pdf，则先处理

        # 获取论文库的论文名称列表
        processed_folder = os.path.join(PROJECT_PATH, "data/processed_papers")
        paper_name_list = os.listdir(processed_folder)

        # 先检查有没有论文，如果没有论文，就直接return
        if len(paper_name_list) == 0:
            return
        else:
            # 如果有论文，则读取并处理
            docs = []
            embeds = []
            for paper_name in paper_name_list:
                # 以防万一先先检查有没有文件，如果没有就跳过
                docs_path = os.path.join(processed_folder, "{}/docs.pkl".format(paper_name))
                embeds_path = os.path.join(processed_folder, "{}/embeds.pkl".format(paper_name))
                if os.path.exists(docs_path) is True and os.path.exists(embeds_path) is True:
                    docs.extend(joblib.load(docs_path))
                    embeds.append(joblib.load(embeds_path))
            embeds = np.vstack(embeds)  # 压平embeds

            # 保存读取的数据
            self.docs = docs
            self.embeds = embeds

            # 构建向量库
            self.index = faiss.IndexFlatL2(embeds.shape[1])
            self.index.add(embeds)

    def vector_based_retrieval(self, query: str, top_k: int) -> list[Document]:
        """
        根据向量相似性获取top_k个documents，、
        :param query: 用户的问题
        :param top_k: 获取最相关documents的数量
        :return: 返回最相关的docs
        """
        # 先检查index是否为None
        if self.index is None:
            return []
        else:
            # 如果数据库存在则正常检索
            query_embed = encode_model.encode([query])   # 对query做embedding

            # 检索并获取最相关的k个embedding的indices
            _, indices = self.index.search(query_embed, top_k)  # 获取top_k的索引
            top_docs = [self.docs[idx] for idx in indices[0]]  # 根据索引提取相关docs

            return top_docs


def rerank_docs(query: str, docs: list[Document], top_k: int) -> list[Document]:
    """
    使用rerank的方式，从docs里找出top_k个与query最相关的docs
    :param query: 用户的问题
    :param docs: 需要rerank的docs
    :param top_k: rerank的top_k
    :return: 返回rerank后的docs
    """
    # rerank
    rerank_input = [[query, doc.page_content] for doc in docs]  # 构建rerank输入
    rerank_score = rerank_model.predict(rerank_input)

    # 根据分数排序，获取top_k的docs
    sorted_idx_list = np.argsort(rerank_score)[::-1]
    top_idx_list = sorted_idx_list[:top_k]
    top_docs = [docs[idx] for idx in top_idx_list]

    return top_docs


def translate_query(query: str) -> str:
    """
    把用户的问题翻译成另一种语言，比如中文变英文，英文变中文。
    :param query: 用户的问题
    :return: 翻译后的query
    """
    # 读取翻译时需要的模型
    translate_model = init_chat_model(model="deepseek-chat", temperature=0.2)

    # 把prompt和query结合
    prompt = f"{translate_prompt}\n\n用户的问题:\n\n{query}"
    messages = [HumanMessage(prompt)]

    # 输入给大模型
    response = translate_model.invoke(messages)

    translated_query = response.content
    return translated_query


def twoWay_retrieval(query: str, vector_top_k: int, rerank_top_k: int):
    """
    采用中英文双路索引的方法，分别召回
    :param query: 用户的问题
    :param vector_top_k: 根据向量召回的top_k
    :param rerank_top_k: rerank的top_k
    :return: 最终retrieval的docs
    """
    # 获取根据vector和bm25检索的docs
    retriever = BaseRetriever()
    vector_docs1 = retriever.vector_based_retrieval(query, vector_top_k)
    vector_docs2 = retriever.vector_based_retrieval(translate_query(query), vector_top_k)
    docs = vector_docs1 + vector_docs2

    # rerank
    top_docs = rerank_docs(query, docs, rerank_top_k)
    return top_docs


if __name__ == "__main__":
    """
    该文件负责检索召回部分
    """
    query = "臭氧的生成机制有哪些"
    docs = twoWay_retrieval(query, 5, 5)
    print(len(docs))

