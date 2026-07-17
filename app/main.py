# FastAPI相关
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.messages import HumanMessage, AIMessage
from fastapi.responses import StreamingResponse

# 项目内部工具
from scripts.agent.paper_agent import paper_agent

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


async def run_agent(question: str, thread_id: str):
    # 构建agent输入
    messages = [HumanMessage(question)]
    config = {"thread_id": thread_id}
    print(type(paper_agent))

    # 获取agent回答
    async for token, metadata in paper_agent.astream(
        input={"messages": messages},
        config={"configurable": config},
        stream_mode="messages",
    ):
        if isinstance(token, AIMessage) and metadata["langgraph_node"] == "model":
            yield token.content


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        run_agent(request.question, "{}_{}".format(request.user_id, request.conversation_id)),
        media_type="text/event-stream",
    )