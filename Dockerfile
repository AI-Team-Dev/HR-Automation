FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl gcc libgl1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Change: use the requirements file inside ats_brain folder
COPY ats_brain/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . /app

EXPOSE 7860

CMD ["uvicorn", "ats_brain.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
