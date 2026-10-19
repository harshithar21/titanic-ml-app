FROM python:3.12-slim
WORKDIR /code
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Re-train inside the image so the saved model always matches the installed scikit-learn version
RUN python -m ml.train --no-plots

EXPOSE 8000
# Render and similar platforms set $PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
