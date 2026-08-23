# Agent相关
from langchain.messages import HumanMessage

# FastAPI相关
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

# 内部工具
from scripts.agent.paper_agent import paper_agent

app = FastAPI()   # 创建FastAPI应用


# 定义request需要的参数
class ChatRequest(BaseModel):
    query: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    """对话接口"""
    # 获取response
    messages = [HumanMessage(request.query)]
    config = {"thread_id": "thread_1"}
    response = paper_agent.invoke(
        input={"messages": messages},
        config={"configurable": config},
    )

    # 从response中获取回答
    answer = response["messages"][-1].content
    return {
        "answer": answer,
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """流式输出接口"""
    # 构建用来获取迭代器的函数
    def generator(request):
        # 获取response
        messages = [HumanMessage(request.query)]
        config = {"thread_id": "thread_1"}
        response = paper_agent.stream(
            input={"messages": messages},
            config={"configurable": config},
            stream_mode="messages",
        )

        # 返回迭代器
        for token, metadata in response:
            if token.content:
                yield token.content

    return StreamingResponse(
        generator(request),
        media_type="text/plain"
    )


if __name__ == "__main__":
    print()
