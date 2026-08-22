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


def encode_all_papers(folder_path: str, flag_reload: bool = False) -> None:
    """
    负责把data/processed_papers中没有embed.pkl的论文进行encode处理
    :param folder_path: paper文件夹所在的位置
    :param flag_reload: 若为True则无论有没有embed.pkl都会处理
    :return: None
    """
    # 获取paper_name_list
    paper_name_list = os.listdir(folder_path)
    for _, paper_name in enumerate(tqdm(paper_name_list, desc="Encode Papers", file=sys.stdout, leave=True)):
        # 构建embeds.pkl路径
        embeds_path = os.path.join(folder_path, "{}/embeds.pkl".format(paper_name))

        # 如果没有或者flag为True，则处理
        docs = joblib.load(os.path.join(folder_path, "{}/docs.pkl".format(paper_name)))
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
            config_path = os.path.join(folder_path, "{}/config_encode.json".format(paper_name))
            with open(config_path, "w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=True, indent=4)


if __name__ == "__main__":
    """
    这个文件负责embedding相关内容
    """
    folder_path = os.path.join(PROJECT_PATH, "data/processed_path")
    encode_all_papers(folder_path, flag_reload=True)
