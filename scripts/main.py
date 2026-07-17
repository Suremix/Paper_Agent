# agent相关
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessageChunk

# 项目内部工具
from agent.paper_agent import paper_agent


if __name__ == "__main__":
    """
    负责和模型进行对话
    """
    while True:
        # 获取输入并判断是否为退出指令
        print("===================================================")
        user_input = input("用户输入:")
        if user_input == "/quit":
            print("程序结束")
            break

        # 构建agent输入
        messages = [HumanMessage(user_input)]
        config = {"thread_id": "thread_1"}

        # 获取agent回答
        response = paper_agent.stream(
            input={"messages": messages},
            config={"configurable": config},
            stream_mode="messages",
        )

        # 打印回答
        print("===================================================")
        print("AI输出:", end="")
        for token, metadata in response:
            if isinstance(token, AIMessageChunk) and metadata["langgraph_node"] == "model":
                print(token.content, end="", flush=True)
        print()
