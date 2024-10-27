FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip3 install setuptools wheel pybind11 build numpy scikit-learn
RUN python3 setup.py build
RUN pip3 install -e .

CMD ["python3", "perf.py"]