# 环境库
from dotenv import load_dotenv

# rewrite相关
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage

# 项目内部工具
from scripts.rag.prompt import rewrite2CN_EN_prompt, translate_prompt

"""读取用于rewrite的模型"""
load_dotenv()   # 读取api

rewrite_model = init_chat_model(model="deepseek-chat")


def rewrite2CN_EN(query: str) -> str:
    """
    把用户的问题翻译成双语，包括中文与英文
    :param query: 用户的问题
    :return: rewrite后的query
    """
    # 把prompt和query结合
    prompt = f"{rewrite2CN_EN_prompt}\n\n用户的问题:\n\n{query}"
    messages = [HumanMessage(prompt)]

    # 输入给大模型
    response = rewrite_model.invoke(messages)

    rewrite_query = response.content
    return rewrite_query


def rewrite_query(query: str) -> str:
    """
    该函数负责整合基于历史对话的rewrite和翻译为中英文的rewrite
    :param query: 用户的问题
    :return: rewrite后的query
    """
    query = rewrite2CN_EN(query)
    return query


def translate_query(query: str) -> str:
    """
    把用户的问题翻译成另一种语言，比如中文变英文，英文变中文。
    :param query: 用户的问题
    :return: 翻译后的query
    """
    # 把prompt和query结合
    prompt = f"{translate_prompt}\n\n用户的问题:\n\n{query}"
    messages = [HumanMessage(prompt)]

    # 输入给大模型
    response = rewrite_model.invoke(messages)

    translated_query = response.content
    return translated_query


if __name__ == "__main__":
    """
    这个文件负责rewrite用户的query
    """

    # test
    history = [
        HumanMessage("臭氧的生成机制包括哪些"),
        AIMessage("1. 氧气在紫外线作用下被分解成氧原子，氧原子再与氧气结合生成臭氧。\n2. 氮氧化物和挥发性有机物在阳光作用下发生光化学反应"),
    ]
    # query = "第二个机制详细介绍一下"
    query = "tell me what happen in the first way"
    rewrite_query = rewrite2CN_EN(query)
    print(rewrite_query)

