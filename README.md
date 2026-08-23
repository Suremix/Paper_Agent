# Paper Agent

Paper Agent是一个针对用户提出的问题，基于本地论文库检索，并根据论文内容回答用户问题的RAG Agent项目。

## Features

- 基于本地论文内容回答用户问题
- 用户可随时手动加入自己搜寻到的论文至本地论文库中
- 可直接在磁盘中删除不希望被检索到的论文

## Installation

### 运行Docker容器开启FastAPI
克隆项目，根据``.env.example``创建``.env``文件
```
git clone https://github.com/Suremix/Paper_Agent.git
cd Paper_Agent
copy .env.example .env
```

在``.env``中填入Deepseek API
```
DEEPSEEK_API_KEY="sk-xxxx"
```

构建容器镜像，并启动容器
```
docker build -t paper_agent .
docker run -v .\data:/app/data --env-file .env -p 8000:8000 paper_agent
```
启动Docker容器后，FastAPI服务将在 http://localhost:8000/ 可用，交互式API文档可访问 http://localhost:8000/docs 。

### 通过uvicorn开启FastAPI
确保``Python>=3.10``，在克隆项目后构建虚拟环境，安装依赖项
```
git clone https://github.com/Suremix/Paper_Agent.git
cd Paper_Agent
python -m venv venv
pip install -r requirements.txt
```
通过uvicorn运行
```
uvicorn cli.fast_api:app --host 0.0.0.0 --port 8000
```


## Paper Management
在``data/papers``中放入原始论文pdf文件，程序处理后的文件将存放在``data/processed_papers``中。
若需要删除某一论文，可手动在``data/processed_papers``中直接删除pdf对应文件夹。
