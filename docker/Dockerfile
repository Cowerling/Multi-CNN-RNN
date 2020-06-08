FROM tensorflow/tensorflow:latest-gpu-py3
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean
RUN apt-get update
RUN apt-get install gdal-bin libgdal-dev -y
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple keras
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rasterio
