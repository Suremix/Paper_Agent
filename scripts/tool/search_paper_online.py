# 标准库
import os

# tool相关
from langchain.tools import tool

# 项目内部工具
from scripts.paper.search_paper import search_paper_by_semantic_scholar


@tool
def search_paper_online(query: str, start_year: int, end_year: int) -> str:
    """
    Search academic papers online based on user requirements.

    Before searching, generate optimized keywords in English from the user's query and use them as the search query.
    Keywords in query should represent broad research concepts, not overly specific descriptions.
    Avoid combining too many constraints into one query.
    If search results are empty or too few:
    - Identify the keyword that contributes the most to narrowing the search scope.
    - Remove or generalize that keyword first.
    - Keep the core research topic unchanged.
    - Repeat this process if necessary until sufficient results are obtained.

    Example:
        Too specific:"deep learning based satellite aerosol optical depth estimation over urban China"
        Relax:"deep learning based satellite aerosol optical depth estimation"
        Further relax:"satellite aerosol optical depth estimation"

    Args:
        query: optimized keywords
        start_year: start year of the paper publication date range
        end_year: end year of the paper publication date range
    """
    # 根据query获取相关论文信息
    # print("\nquery:{} start year:{} end year:{}\n".format(query, start_year, end_year))
    paper_info_list = search_paper_by_semantic_scholar(query, start_year, end_year)

    answer = "The number of papers found: {}\n".format(len(paper_info_list))
    answer += "Paper info:\n\n"
    for paper_info in paper_info_list:
        answer += "title: {}\n".format(paper_info["title"])
        answer += "doi: {}\n".format(paper_info["externalIds"])
        answer += "url: {}\n".format(paper_info["openAccessPdf"]["url"])
        answer += "citation number: {}\n".format(paper_info["citationCount"])
        answer += "abstract: {}\n\n\n".format(paper_info["abstract"])
    return answer


if __name__ == "__main__":
    """
    这个文件负责构建线上搜索论文资料的tool
    """
    # print(search_paper_online("ozone china remote sensing", 2020, 2025))
    print()
