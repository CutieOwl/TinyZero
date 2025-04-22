FROM cybench/kali-linux-large:latest

# Copy the packages list from cybench
COPY cybench/packages.list /tmp/packages.list

# Install common tools, Python 3.9, and Docker
RUN apt-get update && \
    apt-get install -f && \
    apt-get install -y liblzma-dev && \ 
    xargs -a /tmp/packages.list apt-get install -y --no-install-recommends && \
    wget https://www.python.org/ftp/python/3.9.7/Python-3.9.7.tgz && \
    tar xzf Python-3.9.7.tgz && \
    cd Python-3.9.7 && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.9.7 Python-3.9.7.tgz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
    chmod +x /usr/local/bin/docker-compose

WORKDIR /app

RUN ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip3 && \
    python3.9 -m venv /venv

# Copy requirements from TinyZero
RUN /venv/bin/pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121


COPY requirements_noattn.txt /app/
ENV PATH="/venv/bin:$PATH"
RUN /venv/bin/pip install packaging wheel
RUN /venv/bin/pip install -r requirements_noattn.txt

# Copy requirements from cybench
COPY cybench/requirements.txt /app/
RUN /venv/bin/pip install -r requirements.txt
COPY cybench/benchmark /app/benchmark

COPY ./cybench/run_benchmark.py /app/run_benchmark.py
COPY ./cybench/run_task.py /app/run_task.py

# Copy the data preprocessing script and its dependencies
COPY examples/data_preprocess/cybench.py /app/cybench.py

COPY ./cybench/tools/dockerd-entrypoint.sh /usr/local/bin/

COPY cybench/benchmark/requirements.sh /benchmark/requirements.sh
COPY cybench/benchmark/requirements.txt /benchmark/requirements.txt

RUN chmod +x /benchmark/requirements.sh
RUN /benchmark/requirements.sh
RUN /venv/bin/pip install -r /benchmark/requirements.txt

# Copy verl package source code
COPY ./verl /app/verl/
COPY ./README.md /app/README.md

# Copy setup.py
COPY ./setup.py /app/setup.py

# Install verl in development mode
RUN /venv/bin/pip install -e .

RUN mkdir -p /app/data

ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh", "python3", "cybench.py"]