# 标准库
import os
import json
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
model_id = "BAAI/bge-m3"
embed_model_dir = snapshot_download(
    model_id=model_id,
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


def split_docs(docs: list[Document], chunk_size: int = 1024, overlap: int = 200) -> list[Document]:
    """
    该函数负责切分documents文件
    :param docs: 读取的pdf文件的documents
    :param chunk_size: 切分chunk长度
    :param overlap: chunk之间的重叠长度
    :return: 切分之后的documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    docs = splitter.split_documents(docs)
    return docs


def get_docs_embeds(file_path: str, chunk_size: int = 1024, overlap: int = 200) -> [list[Document], np.ndarray]:
    """
    获取pdf路径，切分并获取embeds
    :param file_path: pdf文件路径
    :param chunk_size: 切分chunk长度
    :param overlap: chunk之间的重叠长度
    :return: 切分后的docs与对应的embeds
    """
    docs = load_pdf(file_path)  # 读取pdf
    docs = split_docs(docs, chunk_size, overlap)  # 切分pdf

    texts = [doc.page_content for doc in docs]   # 构建成encode输入格式
    embeds = embed_model.encode_document(texts)   # embedding

    return docs, embeds


def process_1Paper(file_path: str, output_folder: str, chunk_size: int = 1024, overlap: int = 200) -> None:
    """
    获取需要处理的pdf的路径，处理为docs、embeds和metadata后保存在新文件夹中
    :param file_path: 论文路径
    :param output_folder: 各个文件的输出
    :param chunk_size: 切分chunk长度
    :param overlap: chunk之间的重叠长度
    :return: None
    """
    # 读取papers文件夹的pdf文件
    docs, embeds = get_docs_embeds(file_path, chunk_size, overlap)

    # 构建新文件夹
    joblib.dump(docs, os.path.join(output_folder, "docs.pkl"))   # 保存docs
    joblib.dump(embeds, os.path.join(output_folder, "embeds.pkl"))   # 保存embeds

    # 构建metadata，并保存为json形式
    metadata = {
        "split": {
            "method": "RecursiveCharacterTextSplitter",
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
        "docs": {
            "num_chunk": len(docs),
        },
        "embeds": {
            "model": model_id,
            "dimension": embeds.shape[1]
        },
    }

    with open(os.path.join(output_folder, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=4)


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
        paper_name_list.remove(".gitkeep")   # 移除.gitkeep文件

    for i, paper_name in enumerate(paper_name_list):
        # 根据paper_name构建新的论文文件夹，并且剪切pdf
        paper_folder = os.path.join(PROJECT_PATH, "data/processed_papers/{}".format(paper_name))
        os.makedirs(paper_folder, exist_ok=True)

        file_path = os.path.join(folder_path, paper_name)
        new_paper_path = os.path.join(paper_folder, "paper.pdf")
        shutil.move(file_path, new_paper_path)   # 剪切pdf

        # 处理pdf为docs、embeds和metadata
        process_1Paper(new_paper_path, paper_folder)   # 处理pdf
        print("\rProcess {}/{} {} Done".format(i + 1, len(paper_name_list), paper_name), end="")
    print("\rPapers处理完成")


def refresh_paper_library() -> None:
    """
    该函数负责重新构建论文库中所有论文的docs和embeds等文件，在修改切分方法时需要使用
    :return: None
    """
    folder_path = os.path.join(PROJECT_PATH, "data/processed_papers")
    paper_name_list = os.listdir(folder_path)
    if ".gitkeep" in paper_name_list:
        paper_name_list.remove(".gitkeep")   # 移除.gitkeep文件

    for i, paper_name in enumerate(paper_name_list):
        paper_folder = os.path.join(folder_path, paper_name)
        file_path = os.path.join(paper_folder, "paper.pdf")
        process_1Paper(file_path, paper_folder)   # 刷新内容

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
    # process_allPapers()

    """refresh"""
    refresh_paper_library()
