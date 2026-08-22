# 环境库
import os.path
import sqlite3
from dotenv import load_dotenv

# agent相关
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage

# 项目内部工具
from scripts.tool.search_paper_library import search_paper_library
from scripts.agent.prompt import agent_system_prompt
from utils.path import PROJECT_PATH

load_dotenv()   # 加载环境

# 构建agent模型
model = init_chat_model(
    model="deepseek-v4-flash",
    streaming=True,
    temperature=0.5,
)

# 构建checkpointer
checkpoint_folder = os.path.join(PROJECT_PATH, "data/history/")
os.makedirs(checkpoint_folder, exist_ok=True)
conn = sqlite3.connect(
    os.path.join(checkpoint_folder, "chat_history.db"),
    check_same_thread=False
)
checkpointer = SqliteSaver(conn)

# 构建agent
paper_agent = create_agent(
    model=model,
    tools=[search_paper_library],
    system_prompt=agent_system_prompt,
    checkpointer=checkpointer,
    middleware=[
        SummarizationMiddleware(
            model="deepseek-chat",
            trigger=("messages", 50),
            keep=("messages", 20),
        )
    ]
)


if __name__ == "__main__":
    """
    这个文件负责构建agent本体
    """
    config = {"thread_id": "thread_test"}

    while True:
        user_input = input("用户输入:")

        messages = [HumanMessage(user_input)]
        response = paper_agent.invoke(
            input={"messages": messages},
            config={"configurable": config},
        )

        for message in response["messages"]:
            message.pretty_print()
        print()

