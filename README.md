# Paper Agent

Paper Agent是一个针对用户提出的问题，基于本地论文库检索，并根据论文内容回答用户问题的RAG Agent项目。
除了基本的RAG系统外，项目还预计为Agent设计了其他tools，比如当本地论文库中的知识不涉及或无法回答用户的问题时，能够在网上搜索论文，并根据论文摘要筛选、下载论文的功能。
目前只实现了基于Semantic Scholar API检索论文的tool，其他功能仍在尝试实现中。

## Features

- 基于本地论文内容回答用户问题
- 用户可随时手动加入自己搜寻到的论文至本地论文库中
- 可直接在磁盘中删除不希望被检索到的论文
- 基于Semantic Scholar的论文搜索功能

## Installation

### **运行Docker容器以获取FastAPI（面向用户）**
克隆项目，进入项目文件夹，构建Docker镜像：
```
git clone https://github.com/Suremix/Paper_Agent.git
cd Paper_Agent
docker build -t paper_agent .
```

创建自己的项目文件夹，然后运行以下命令:
```
docker run -p 8000:8000 paper_agent
```

重命名``.env.example``文件为``.env``，并填入你的Deepseek API Key.
```
cd app
mv .env.example .env
```

启动Docker容器后，FastAPI服务将在 http://localhost:8000/ 可用，交互式API文档可访问 http://localhost:8000/docs 。
