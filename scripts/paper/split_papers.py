# 标准库
import os
import sys
import json
import joblib
from tqdm import tqdm

# 切分相关
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 注释类
from langchain_core.documents import Document

# 内部
from utils.path import PROJECT_PATH
from configs.config import configs


def load_paper_md(file_path: str) -> str:
    """
    负责读取md形式的paper
    :param file_path: 文件路径
    :return: 返回论文内容
    """
    with open(file_path, "r") as file:
        paper_md = file.read()
    return paper_md


def split_paper(paper_md: str, chunk_size: int, overlap: int) -> list[Document]:
    """
    负责对md格式的论文进行切分。
    思路是先对header进行切分，然后再对长度过长的chunk进一步切分
    :param paper_md:
    :param chunk_size: 每个chunk的大小
    :param overlap: chunk之间的重叠区域
    :return:
    """
    # 先对markdown的header进行切分
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header1"),
            ("##", "header2"),
        ]
    )
    header_split_docs = header_splitter.split_text(paper_md)

    # 找出长度过长的chunk，根据###以及换行进行再切分
    # 这里不对空格进行切分是为了保证完整性
    para_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=[
            "\n###",
            "\n\n",
            "\n",
        ]
    )

    para_split_docs = []
    for doc in header_split_docs:
        # 检查长度是否超过chunk_size
        if len(doc.page_content) > chunk_size:
            sub_docs = para_splitter.split_documents([doc])
            para_split_docs += sub_docs
        else:
            para_split_docs.append(doc)

    # 如果仍然不行，则再按照空格进行切分
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n###",
            "\n\n",
            "\n",
            "。",
            ".",
            " ",
            "",
        ]
    )
    char_split_docs = []
    for doc in para_split_docs:
        if len(doc.page_content) > chunk_size:
            sub_docs = char_splitter.split_documents([doc])
            char_split_docs += sub_docs
        else:
            char_split_docs.append(doc)

    # 返回
    docs = char_split_docs
    return docs


def split_all_papers(folder_path: str, flag_reload: bool = False) -> None:
    """
    该函数会对data/processed_papers的所有文件夹遍历，对没有docs.pkl的论文进行处理。
    :param folder_path: paper文件夹所在位置
    :param flag_reload: 若为True，则即便docs.pkl已存在，也会进行切分并覆盖
    :return:
    """
    # 获取paper_name_list并遍历
    paper_name_list = os.listdir(folder_path)
    for _, paper_name in enumerate(tqdm(paper_name_list, desc="Split Papers", file=sys.stdout, leave=True)):
        # 构建docs.pkl的路径
        docs_path = os.path.join(folder_path, "{}/docs.pkl".format(paper_name))

        # 如果没有docs.pkl，或者flag_reload为True，则对该paper进行切分
        chunk_size = configs["split"]["chunk_size"]
        overlap = configs["split"]["overlap"]
        if os.path.exists(docs_path) is False or flag_reload is True:
            md_path = os.path.join(folder_path, "{}/paper.md".format(paper_name))
            paper_md = load_paper_md(md_path)
            docs = split_paper(paper_md, chunk_size, overlap)

            # 给每个doc的metadata加上source这一项
            for doc in docs:
                doc.metadata["source"] = paper_name

            # 根据docs.pkl路径保存切分结果
            joblib.dump(docs, docs_path)

            # 保存config信息
            len_list = [len(doc.page_content) for doc in docs]
            config = {
                "num_chunk": len(docs),
                "max_chunk": max(len_list),
                "min_chunk": min(len_list),
                "mean_chunk": round(sum(len_list) / len(len_list), 2),
                "chunk_size": chunk_size,
                "overlap": overlap,
            }
            config_path = os.path.join(folder_path, "{}/config_split.json".format(paper_name))
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    """
    这个文件负责文档切分的内容
    """
    print()
    folder_path = os.path.join(PROJECT_PATH, "data/processed_papers")
    split_all_papers(folder_path, flag_reload=True)
    paper_name = "Allen 等 - 2012 - Recent Northern Hemisphere tropical expansion prim"
    docs = joblib.load(os.path.join(PROJECT_PATH, "data/processed_papers/{}/docs.pkl".format(paper_name)))
    for doc in docs:
        print(doc.metadata)
