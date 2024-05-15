FROM waggle/plugin-base:1.1.1-ml-tensorflow2.3-arm64
WORKDIR /app
ENV MODEL /app/flood_detection_model.keras
COPY /app /app
CMD ["python", "app.py"]