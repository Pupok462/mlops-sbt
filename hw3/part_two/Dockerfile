FROM nvcr.io/nvidia/tritonserver:23.12-py3

RUN pip install torch torchvision transformers onnxruntime

COPY assets /assets
COPY model_repository /models

EXPOSE 8000 8001 8002

#CMD ["tritonserver", "--model-repository=/models"]

# sudo docker run --gpus=all -d --rm -p8000:8000 -p8001:8001 -p8002:8002 pupok
# docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.12-py3-sdk bash
