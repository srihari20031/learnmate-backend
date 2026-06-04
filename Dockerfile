#always start with a base image 
FROM python:3.12-slim
# set the working directory in the container
WORKDIR /app
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
