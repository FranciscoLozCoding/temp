FROM waggle/plugin-tensorflow:2.0.0-cuda-l4t
WORKDIR /app
ENV MODEL /app/flood_detection_model.keras
COPY /app /app
RUN pip install -r ./requirements.txt
CMD ["python", "app.py"]