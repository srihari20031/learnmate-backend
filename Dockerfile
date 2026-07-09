#always start with a base image 
FROM python:3.12-slim
# set the working directory in the container
WORKDIR /app

# Install torch FIRST, from PyTorch's CPU-only index.
#
# sentence-transformers pulls in torch, and on linux/amd64 a plain `pip install torch`
# resolves to the CUDA build: ~2.5GB of nvidia-* wheels this container will never use
# (there is no GPU). Installing the CPU wheel up front means the requirements.txt step
# below already sees torch as satisfied and won't reach for the CUDA one.
#
# It also sits above the COPY, so Docker caches this layer across code-only changes.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

# copy the requirements file into the container
COPY requirements.txt .
# install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# copy the rest of the application code into the container
COPY . .
# expose the port that the application will run on
EXPOSE 8000
# command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
