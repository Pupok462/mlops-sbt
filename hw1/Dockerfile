FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip3 install setuptools wheel pybind11 build numpy scikit-learn
RUN python3 -m build
RUN pip3 install dist/pupok-0.1-py3-none-any.whl

CMD ["python3", "perf.py"]