# 标准库
import os
import sys
import json
import joblib
from tqdm import tqdm

# embedding相关
import torch
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

# 内部
from utils.path import PROJECT_PATH

# 导入embed模型
model_id = "BAAI/bge-m3"
encode_model_dir = snapshot_download(
    model_id=model_id,
    cache_dir=os.path.join(PROJECT_PATH, "models")  # 将模型下载至项目中
)
device = "cuda" if torch.cuda.is_available() else "cpu"   # 检查是否可以使用cuda
encode_model = SentenceTransformer(encode_model_dir, device=device)


def encode_all_papers(processed_folder: str, flag_reload: bool = False, flag_show_text: bool = False) -> None:
    """
    负责把data/processed_papers中没有embed.pkl的论文进行encode处理
    :param processed_folder: paper文件夹所在文件夹
    :param flag_reload: 若为True则无论有没有embed.pkl都会处理
    :param flag_show_text: 是否显示信息
    :return: None
    """
    # 获取paper_name_list
    paper_name_list = os.listdir(processed_folder)
    # 根据信号决定是否显示进度条
    if flag_show_text is True:
        enu = tqdm(paper_name_list, desc="Split Papers", file=sys.stdout, leave=True)
    else:
        enu = paper_name_list
    # 遍历所有paper文件夹
    for _, paper_name in enumerate(enu):
        # 构建embeds.pkl路径
        embeds_path = os.path.join(processed_folder, "{}/embeds.pkl".format(paper_name))

        # 如果没有或者flag为True，则处理
        docs = joblib.load(os.path.join(processed_folder, "{}/docs.pkl".format(paper_name)))
        if os.path.exists(embeds_path) is False or flag_reload is True:
            # 构建texts，扔给embedding模型
            texts = [doc.page_content for doc in docs]
            embeds = encode_model.encode_document(texts)  # embedding

            # 根据embed路径保存
            joblib.dump(embeds, embeds_path)

            # 保存config信息
            config = {
                "embedding model": model_id,
                "dimension": embeds.shape[1],
            }
            config_path = os.path.join(processed_folder, "{}/config_encode.json".format(paper_name))
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    """
    这个文件负责embedding相关内容
    """
    folder_path = os.path.join(PROJECT_PATH, "data/processed_path")
    encode_all_papers(folder_path, flag_reload=True)
