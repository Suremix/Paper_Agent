# Tool相关
from langchain.tools import tool

# 项目内部工具
from scripts.rag.retrieval import retrieval_docs
from scripts.rag.rewrite import rewrite_query
from scripts.rag.prompt import write_rag_prompt


@tool
def read_paper_library(rewritten_query: str) -> str:
    """
    Search academic papers to answer research questions.
    Before calling this tool, convert the user's question into a standalone search query.
    The input should not contain unresolved references such as "it", "this", or "that".

    Example:
        raw query: What are the influencing factors.
        rewritten query: What are the influencing factors of ozone (rewrite base on conversation)

    Args:
        rewritten_query: rewritten standalone query
    """
    rewritten_query = rewrite_query(rewritten_query)   # 重写query
    top_docs = retrieval_docs(rewritten_query)  # 用重写过后的query检索
    docs_prompt = write_rag_prompt(top_docs)  # 把docs组成prompt
    return docs_prompt


if __name__ == "__main__":
    """
    这个文件负责构建一个阅读论文库的论文，并通过RAG回答用户问题的tool
    """
    # docs_prompt = read_paper_library("帮我找一下2025年中国臭氧浓度反演的文章有哪些")
    # print(docs_prompt)
    print()
