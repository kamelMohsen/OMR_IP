FROM continuumio/miniconda3

WORKDIR /app

COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
# Create the environment:
RUN conda env create -n myenv --file=requirements.yml

ENTRYPOINT /opt/conda/bin/conda run -n myenv python main.py $INPUT_TEST $OUTPUT_TEST

# ENTRYPOINT ["tail", "-f", "/dev/null"]
