# 标准库
import os
import sys
import json
import time
import shutil
import logging
from tqdm import tqdm

# api相关
import requests
import subprocess

# 注释类
from typing import Literal
from enum import Enum

# 内部
from utils.path import PROJECT_PATH
from configs.config import configs

# 设置环境变量
os.environ["MINERU_MODEL_SOURCE"] = "local"   # 设置调用为本地

# 设置temp文件夹位置
temp_folder = os.path.join(PROJECT_PATH, "data/temp")
os.makedirs(temp_folder, exist_ok=True)
os.environ["MINERU_API_OUTPUT_ROOT"] = temp_folder

# 创建logger对象
logger = logging.getLogger(__name__)


class MinerU_Client:
    def __init__(self, host: str, port: str, num_retry: int = 5, sleep: int = 3):
        """
        client初始化
        :param host: 监听地址
        :param port: 监听端口
        :param num_retry: 最大重连次数
        :param sleep: 操作间隔时间
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.num_retry = num_retry
        self.sleep = sleep
        self.process = None

    def start(self) -> bool:
        """
        该函数负责等待API服务启动
        :return: 返回bool来表示是否成功start
        """
        self.process = subprocess.Popen([
            "mineru-api",
            "--host",
            self.host,
            "--port",
            self.port
        ])
        time.sleep(self.sleep)

        # 等待服务启动
        url = f"{self.base_url}/health"   # 检测接口
        for i in range(0, self.num_retry):
            # 尝试启动，如果遇到报错，则在显示报错之前关闭client
            try:
                response = requests.get(url=url)
                # 如果状态码是200，则启动成功，否则休眠一段时间，然后进入下一次尝试
                if response.status_code == 200:
                    return True
                else:
                    logger.warning(
                        "Mineru启动失败，重试中(%d/%d)，失败原因: 状态码不为200（实际为%d）",
                        i + 1, self.num_retry, response.status_code,
                    )
                    time.sleep(self.sleep)
                    continue
            except requests.exceptions.ConnectionError as e:
                # 如果是ConnectionError，可能是还没启动好，休眠一段时间再继续
                logger.warning(
                    "Mineru启动失败，重试中(%d/%d)，失败原因: 存在错误->%s",
                    i + 1, self.num_retry, e,
                )
                time.sleep(self.sleep)
            except Exception as e:
                # 如果是其他Exception，则先关闭client，再抛出异常
                self.stop()
                raise e

        # 如果尝试次数已达最大重试次数，则判断为启动失败
        self.stop()
        return False

    def stop(self) -> None:
        """
        该函数负责关闭api进程
        :return: None
        """
        # 检查process是否是None，若不是再关闭
        if self.process is not None:
            self.process.terminate()
            self.process.wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def parse(self, pdf_path, backend: str = "hybrid-engine", effort: str = "medium") -> dict:
        """
        负责调用mineru的file_parse去分析pdf
        :param pdf_path: pdf文件路径
        :param backend: mineru采用的处理方式
            - "pipeline": 普通的pipeline模型
            - "vlm-engine": 采用视觉模型
            - "hybrid-engine": pipeline与视觉模型结合
        :param effort: 若backend为hybrid-engine的时候的模型强度
            - "medium": 更快的速度，但效果没那么好
            - "high": 更好的效果，但速度更慢
        :return: 返回response的json格式结果（字典格式）
        """
        # 提前设置好后面post要传的参数
        files = {"files": open(pdf_path, "rb")}
        data = {
            "return_md": "true",
            "return_content_list": "true",
            "backend": backend,
        }
        if backend == "hybrid-engine":
            data["effort"] = effort

        # request
        url = f"{self.base_url}/file_parse"   # 分析接口
        for _ in range(0, self.num_retry):
            response = requests.post(url=url, files=files, data=data)
            # 如果状态码是200，则获取成功，否则休眠一段时间，然后进入下一次尝试
            if response.status_code == 200:
                return response.json()

            # 如果状态码不是200，就休眠一下
            time.sleep(self.sleep)

        # 如果这个循环跑出来了，说明状态码一直不是200，就输出一下情况，然后返回个空字典
        print("错误: Parse Paper无法连接成功")
        return {}


def build_paper_folder(file_folder: str, output_folder: str, flag_delete_file: bool = True) -> None:
    """
    该函数负责为每个论文pdf文件创建文件夹，并将pdf复制到该文件夹下。
    :param file_folder: 原论文pdf所存放的文件夹
    :param output_folder: paper文件夹需要存放的位置
    :param flag_delete_file: 是否在创建好文件夹后删除papers里的文件
    :return: None
    """
    # 扫描file_folder文件夹中的文件，获取pdf名称列表（需要检查是否有.gitkeep文件并处理）
    file_name_list = os.listdir(file_folder)
    if ".gitkeep" in file_name_list:
        file_name_list.remove(".gitkeep")   # 移除.gitkeep文件

    # 遍历列表，在output_folder中创建文件夹，并移动pdf
    os.makedirs(output_folder, exist_ok=True)   # 确保输出文件夹存在
    for i, file_name in enumerate(tqdm(file_name_list, desc="Build Paper Folder", file=sys.stdout, leave=True)):
        # 获取无后缀处理空格后的pdf的名字
        paper_name = os.path.splitext(file_name)[0].strip(" ")

        # 创建文件夹
        paper_folder = os.path.join(output_folder, paper_name)
        os.makedirs(paper_folder, exist_ok=True)

        # 移动文件并更名为paper.pdf
        file_path = os.path.join(file_folder, file_name)  # 原文件路径
        new_paper_path = os.path.join(paper_folder, "paper.pdf")  # 新文件路径

        if flag_delete_file is True:
            shutil.move(file_path, new_paper_path)  # 剪切pdf
        else:
            shutil.copy(file_path, new_paper_path)


def clean_temp_folder() -> None:
    """
    该函数负责清除temp文件夹中的临时文件
    :return: None
    """
    for file_name in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file_name)
        # 如果是文件就直接删除，如果是文件夹就用rmtree
        if os.path.isfile(file_path) is True:
            os.remove(file_path)
        elif os.path.isdir(file_path) is True:
            shutil.rmtree(file_path)


def pares_papers(folder_path: str) -> None:
    """
    该函数负责把processed_papers里，没有paper.md与contents.json的论文处理为资料
    :param folder_path: paper文件夹路径
    :return: None
    """
    # 获取paper_name_list，然后遍历
    paper_name_list = os.listdir(folder_path)
    need_list = []
    # 先看有没有paper.md和contents.pkl，如果没有，就存到一个列表need_list
    for paper_name in paper_name_list:
        md_path = os.path.join(folder_path, "{}/paper.md".format(paper_name))
        contents_path = os.path.join(folder_path, "{}/contents.json".format(paper_name))

        if os.path.exists(md_path) is False or os.path.exists(contents_path) is False:
            need_list.append(paper_name)

    # 等遍历完了之后，如果need_list长度大于0，说明有需要处理的paper，创建client
    if len(need_list) > 0:
        host = configs["mineru"]["host"]
        port = str(configs["mineru"]["port"])

        # 打开client
        with MinerU_Client(host, port) as client:
            # 设置try-except防止报错时没有关闭client
            for _, paper_name in enumerate(tqdm(need_list, desc="Pares Papers", file=sys.stdout, leave=True)):
                # 把pdf_path输入给client，获取response
                pdf_path = os.path.join(folder_path, "{}/paper.pdf".format(paper_name))
                response = client.parse(pdf_path)

                # 从response中获取results对应的字典，再把md_content和content_list拿出来
                result = response["results"][response["file_names"][0]]
                paper_md = result["md_content"]
                contents = json.loads(result["content_list"])

                # 分别按照paper.md和contents.pkl的路径保存
                # 保存md
                md_path = os.path.join(folder_path, "{}/paper.md".format(paper_name))
                with open(md_path, "w", encoding="utf-8") as file:
                    file.write(paper_md)

                contents_path = os.path.join(folder_path, "{}/contents.json".format(paper_name))
                with open(contents_path, "w", encoding="utf-8") as file:
                    json.dump(contents, file, ensure_ascii=True, indent=4)

        # 跑完要处理临时文件
        clean_temp_folder()


if __name__ == "__main__":
    """
    这个文件负责使用读取论文pdf文件，并为每个pdf创建单独的文件夹，存储原始pdf和读取后的资料（如md文件等），
    待处理论文需放在data/papers文件夹中，而处理后的资料将放在data/processed_papers中
    """
    file_folder = os.path.join(PROJECT_PATH, "data/papers/")
    output_folder = os.path.join(PROJECT_PATH, "data/processed_papers/")

    build_paper_folder(file_folder, output_folder, flag_delete_file=False)
    pares_papers(output_folder)

    # client = MinerU_Client("127.0.0.1", "8000")
    # client.start()
    # s = input()
    # client.stop()

