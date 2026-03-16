FROM python:3.9-slim
WORKDIR /home
RUN mkdir data_files
COPY docker_pyt /home
RUN pip install -r requirements.txt
CMD ["python", "RAG.py"]