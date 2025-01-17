FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y build-essential

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8501

CMD ["streamlit", "run", "_1_Introduction_tidee.py"] 
#"--server.port 8501", "--server.adress 0.0.0.0", "server.enableCORS=false", "--server.enableXsrfProtection=false"]