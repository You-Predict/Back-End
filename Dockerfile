FROM python:3.6.12

# set a directory for the app
ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . /app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# define the port number the container should expose
EXPOSE 5000/tcp
EXPOSE 5000/udp
# run the command
CMD ["python3", "API.py"]
