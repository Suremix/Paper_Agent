# 环境库
from dotenv import load_dotenv

# agent相关
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage

# 项目内部工具
from scripts.tool.read_paper_library import read_paper_library
from scripts.tool.search_paper_online import search_paper_online
from scripts.agent.prompt import agent_system_prompt

load_dotenv()   # 加载环境
model = init_chat_model("deepseek-chat", streaming=True)

paper_agent = create_agent(
    model=model,
    tools=[read_paper_library, search_paper_online],
    system_prompt=agent_system_prompt,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="deepseek-chat",
            trigger=("messages", 20),
            keep=("messages", 6),
        )
    ]
)


if __name__ == "__main__":
    """
    这个文件负责构建agent本体
    """

    while True:
        user_input = input("用户输入:")

        messages = [HumanMessage(user_input)]
        config = {"thread_id": "thread_1"}

        response = paper_agent.invoke(
            input={"messages": messages},
            config={"configurable": config},
        )

        for message in response["messages"]:
            message.pretty_print()
        print()

