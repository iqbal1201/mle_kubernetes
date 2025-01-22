FROM python:3.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
# RUN pip install gunicorn
RUN pip install -r requirements.txt

# Expose port 
ENV PORT=8080

# Run the application:
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--worker-class", "sync" , "--config=config.py"]
