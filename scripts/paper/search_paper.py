# 标准库
import os
import time
import requests

# 环境库
from dotenv import load_dotenv

load_dotenv()   # 读取.env文件


def search_paper_by_semantic_scholar(
    query: str,
    start_year: int,
    end_year: int,
    max_retry_time: int = 10,
    retry_wait: int = 5,
    max_paper_num: int = 100,
) -> list[dict]:
    """
    该函数负责按照关键词query，去使用semantic scholar搜索start_year至end_year之间的相关论文，并返回相关信息。
    :param query: 关键词
    :param start_year: 开始年份
    :param end_year: 结束年份
    :param max_retry_time: 无法响应时，重试的次数
    :param retry_wait: retry的时候等待间隔
    :param max_paper_num: 最大论文下载数量
    :return: 返回论文信息列表
    """
    # 设定request所需的信息
    url = os.getenv("SEMANTIC_SCHOLAR_BASE_URL")
    params = {
        "query": query,
        "year": "{}-{}".format(start_year, end_year),
        "fields": "title,openAccessPdf,abstract,citationCount,externalIds",
    }

    # 开始请求，获取相关论文信息
    response_data = {}
    for i in range(1, max_retry_time + 1):
        response = requests.get(url, params)
        # 如果请求成功，则修改flag，并跳出循环
        if response.status_code == 200:
            response_data = response.json()
            break
        else:
            # 如果没有请求成功，则等待一段时间再次请求
            # print("Semantic Scholar 请求失败，status code: {}，重试第{}次".format(response.status_code, i))
            time.sleep(retry_wait)

    # 把论文信息整理为列表
    paper_list = []
    # 先检查response_data是否为空
    if len(response_data["data"]) != 0:
        # 若不为空，则筛选出可下载的论文信息
        for paper_info in response_data["data"]:
            if paper_info["openAccessPdf"]["status"] == "GOLD":
                paper_list.append(paper_info)

    # 限制长度
    if len(paper_list) > max_paper_num:
        paper_list = paper_list[:max_paper_num]

    # print("查找到{}篇论文，可下载论文数量为: {}".format(response_data["total"], len(paper_list)))
    return paper_list


if __name__ == "__main__":
    """
    该文件负责构建用于论文搜索的函数。
    """
    paper_info_list = search_paper_by_semantic_scholar(
        query="ozone remote sensing China",
        start_year=2024,
        end_year=2025,
    )
    for paper_info in paper_info_list:
        print(paper_info)
