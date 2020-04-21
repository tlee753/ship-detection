FROM python:3.7
COPY ship-detect-app /app
WORKDIR /app
RUN pip install -r requirements.txt
COPY visualize.py /usr/local/lib/python3.7/site-packages/mrcnn/visualize.py
# ENTRYPOINT ["python", "ship-infer-8.py"]
