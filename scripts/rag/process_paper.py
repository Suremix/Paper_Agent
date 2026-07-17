# 标准库
import os
import shutil
import joblib

# 科学计算库
import numpy as np

# 数据处理相关
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

# 注释类
from langchain_core.documents import Document

# 项目内部工具
from utils.path import PROJECT_PATH


# 提前准备好embedding的模型
embed_model_dir = snapshot_download(
    model_id="BAAI/bge-m3",
    cache_dir=os.path.join(PROJECT_PATH, "models")  # 将模型下载至项目中
)
embed_model = SentenceTransformer(embed_model_dir, device="cuda")


def load_pdf(file_path: str) -> list[Document]:
    """
    该函数负责读取pdf文件
    :param file_path: pdf文件路径
    :return: 包含每页document对象的list
    """
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_docs(docs: list[Document]) -> list[Document]:
    """
    该函数负责切分documents文件
    :param docs: 读取的pdf文件的documents
    :return: 切分之后的documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = splitter.split_documents(docs)
    return docs


def get_docs_embeds(file_path: str) -> [list[Document], np.ndarray]:
    """
    获取pdf路径，切分并获取embeds
    :param file_path: pdf文件路径
    :return: 切分后的docs与对应的embeds
    """
    docs = load_pdf(file_path)  # 读取pdf
    docs = split_docs(docs)  # 切分pdf

    texts = [doc.page_content for doc in docs]   # 构建成encode输入格式
    embeds = embed_model.encode_document(texts)   # embedding

    return docs, embeds


def process_1Paper(paper_name: str) -> None:
    """
    对于papers文件夹中的论文，在processed_papers中创建文件夹，移动并改名到新文件夹中，处理为docs和embeds后保存在新文件夹中
    :param paper_name: 论文名称
    :return: None
    """
    # 读取papers文件夹的pdf文件
    folder_path = os.path.join(PROJECT_PATH, "data/papers")
    file_path = os.path.join(folder_path, paper_name)
    docs, embeds = get_docs_embeds(file_path)

    # 构建新文件夹
    paper_folder = os.path.join(PROJECT_PATH, "data/processed_papers/{}".format(paper_name))
    os.makedirs(paper_folder, exist_ok=True)

    shutil.move(file_path, os.path.join(paper_folder, "paper.pdf"))   # 剪切pdf
    joblib.dump(docs, os.path.join(paper_folder, "docs.pkl"))   # 保存docs
    joblib.dump(embeds, os.path.join(paper_folder, "embeds.pkl"))   # 保存embeds


def process_allPapers() -> None:
    """
    该函数负责处理papers文件夹中所有未处理的pdf文件，包括整理pdf与构建docs、embeds
    在processed_papers文件夹中存放处理后的文件
    :return: None
    """
    # 遍历papers文件夹
    folder_path = os.path.join(PROJECT_PATH, "data/papers")
    paper_name_list = os.listdir(folder_path)
    if ".gitkeep" in paper_name_list:
        paper_name_list.remove(".gitkeep")   # 提出.gitkeep文件

    for i, paper_name in enumerate(paper_name_list):
        if paper_name == ".gitkeep":
            continue

        process_1Paper(paper_name)   # 处理pdf
        print("\rProcess {}/{} {} Done".format(i + 1, len(paper_name_list), paper_name), end="")
    print("\rPapers处理完成")


def refresh_paper_library() -> None:
    """
    该函数负责重新构建论文库中所有论文的docs和embeds，在修改切分方法时需要使用
    :return: None
    """
    folder_path = os.path.join(PROJECT_PATH, "data/processed_papers")
    paper_name_list = os.listdir(folder_path)
    for i, paper_name in enumerate(paper_name_list):
        paper_folder = os.path.join(folder_path, paper_name)
        file_path = os.path.join(paper_folder, "paper.pdf")

        docs, embeds = get_docs_embeds(file_path)
        joblib.dump(docs, os.path.join(paper_folder, "docs.pkl"))   # 更新docs
        joblib.dump(embeds, os.path.join(paper_folder, "embeds.pkl"))   # 更新embeds

        print("\rRefresh {}/{} {} Done".format(i + 1, len(paper_name_list), paper_name), end="")
    print("\r论文库刷新完成")


if __name__ == "__main__":
    """
    这个文件负责把下载的论文全部处理为documents和embedding，
    每个论文的初始形态是，在paper文件夹中每个论文各对应一个文件夹，文件夹的名称为原pdf的名称，
    每个论文的pdf文件在文件夹下的名称均为paper.pdf
    """

    """load split"""
    # docs = load_pdf("../../data/papers/Allen 等 - 2012 - Recent Northern Hemisphere tropical expansion prim.pdf")
    # print(docs)
    # print(type(docs), type(docs[0]), len(docs))
    # docs = split_docs(docs)
    # print(len(docs))

    """process"""
    process_allPapers()

    """refresh"""
    # refresh_paper_library()
