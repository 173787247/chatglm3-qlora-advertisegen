# 使用与其他成功项目一致的 PyTorch CUDA 镜像（已验证兼容 RTX-5080）
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

WORKDIR /app

# 设置环境变量（与其他成功项目一致）
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=8

# 配置 Ubuntu apt 使用清华镜像源（加速下载，避免网络问题）
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true

# 安装系统依赖（包括编译 bitsandbytes 需要的工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 复制 requirements 文件（如果有）
COPY requirements.txt* ./

# 设置 pip 使用清华镜像源（加速下载，与其他项目一致）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 初始化 git lfs（用于下载大模型）
RUN git lfs install

# 安装 Python 依赖（使用清华镜像源）
RUN pip install --no-cache-dir \
    transformers==4.37.2 \
    datasets==2.16.1 \
    accelerate==0.26.1 \
    peft==0.7.1 \
    pandas==2.1.1 \
    python-dotenv \
    sentencepiece \
    protobuf \
    cpm-kernels \
    mdtex2html \
    gradio \
    scikit-learn==1.3.2 \
    evaluate==0.4.1

# 安装 bitsandbytes 构建依赖
RUN pip install --no-cache-dir scikit-build-core ninja packaging

# 从源码编译 bitsandbytes for CUDA 12.8
RUN git clone https://github.com/TimDettmers/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    CUDA_VERSION=128 BNB_CUDA_VERSION=128 pip install . --no-build-isolation && \
    cd / && rm -rf /tmp/bitsandbytes

# 设置 bitsandbytes 环境变量（兼容 CUDA 12.8）
ENV BITSANDBYTES_NOWELCOME=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:${LD_LIBRARY_PATH}

# 复制项目文件
COPY . .

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

CMD ["python", "train_chatglm3_advertise.py", "--help"]

