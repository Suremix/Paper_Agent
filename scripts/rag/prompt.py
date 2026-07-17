# 标准库
import os

# 注释类
from langchain_core.documents import Document


rewrite2CN_EN_prompt = """
# 身份：
- 你是一个问题翻译助手，你负责把用户提出的问题变为中英双语的形式。

# 指令：
- 假设用户的问题为中文，则翻译为英文后，中文原文的下方一行放上英文翻译。
- 假设用户的问题为英文，则翻译为中文后，在中文翻译的下方一行放上英文原文。
- 不要回答用户的问题。

# 示例：
user: 臭氧的生成机制是什么
assistant: 
臭氧的生成机制是什么
What is the mechanism of ozone generation?

user: What factors affect ozone concentration?
影响臭氧浓度的因素有哪些？
What factors affect ozone concentration?
"""


def write_rag_prompt(docs: list[Document]) -> str:
    """
    该函数负责获取rag_tool中获取的相关docs构建prompt
    :param docs: 与问题最相关的docs
    :return: 组合的prompt
    """
    # 遍历doc把相关信息写进prompt
    prompt = "下面是相关论文内容:\n\n"
    for doc in docs:
        content = doc.page_content
        page = doc.metadata["page"]
        source = doc.metadata["source"]
        source = os.path.basename(os.path.dirname(source))   # 获取论文文件夹名字

        prompt += f"[文献来源：{source}，页码: {page}]\n{content}\n\n"

    return prompt


if __name__ == "__main__":
    """
    这个文件夹存放RAG文件夹中代码所需要的prompt
    """
    print()