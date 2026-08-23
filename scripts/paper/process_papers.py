# 标准库
import os

# 内部
from scripts.paper.load_papers import build_paper_folder, pares_papers
from scripts.paper.split_papers import split_all_papers
from scripts.paper.encode_papers import encode_all_papers

from utils.path import PROJECT_PATH


def process_papers(
    flag_reload_split: bool = False,
    flag_reload_encode: bool = False,
    flag_show_text: bool = False
) -> None:
    """
    负责对data/papers里的论文进行处理，包括读取markdown、切分以及embedding操作。
    :param flag_reload_split: 是否强制重新切分
    :param flag_reload_encode: 是否强制重新encode
    :param flag_show_text: 是否显示信息
    :return: None
    """
    # 先检查data/papers是否有未被处理的pdf，如果有，就创建新文件夹，然后用mineru处理为md
    pdf_folder = os.path.join(PROJECT_PATH, "data/papers")
    processed_folder = os.path.join(PROJECT_PATH, "data/processed_papers")
    os.makedirs(processed_folder, exist_ok=True)   # 创建存储paper文件夹的文件夹

    # 读取pdf列表
    file_name_list = os.listdir(pdf_folder)
    if ".gitkeep" in file_name_list:
        file_name_list.remove(".gitkeep")   # 移除.gitkeep文件

    # 如果存在未处理的pdf，则需要先创建文件夹
    if len(file_name_list) > 0:
        build_paper_folder(pdf_folder, processed_folder, flag_show_text=flag_show_text)
        pares_papers(processed_folder, flag_show_text=flag_show_text)

    # 然后切分、embedding
    split_all_papers(processed_folder, flag_reload=flag_reload_split, flag_show_text=flag_show_text)
    encode_all_papers(processed_folder, flag_reload=flag_reload_encode, flag_show_text=flag_show_text)


if __name__ == "__main__":
    process_papers(flag_show_text=True)


