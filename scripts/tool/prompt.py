# 标准库
import os

# 注释类
from langchain_core.documents import Document


def build_docs_prompt(docs: list[Document]) -> str:
    """
    该函数负责将从rag获取的docs组合为提示词，供agent使用
    :param docs: 与问题最相关的docs
    :return: 组合的prompt
    """
    # 遍历doc把相关信息写进prompt
    prompt = "下面是相关论文内容:\n\n"
    for doc in docs:
        # 获取内容、来源
        content = doc.page_content
        source = doc.metadata["source"]

        # 获取section
        if "Header2" in doc.metadata:
            section = doc.metadata["header2"]
        else:
            section = doc.metadata["header1"]

        prompt += f"[Source：{source}, Section：{section}]\n{content}\n\n"

    return prompt


if __name__ == "__main__":
    """
    这个文件夹存放RAG文件夹中代码所需要的prompt
    """
    print()
