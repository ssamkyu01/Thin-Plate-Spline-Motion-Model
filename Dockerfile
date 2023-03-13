FROM python:3.9

WORKDIR /app
COPY . .

RUN python3 -m pip install --upgrade pip
RUN apt-get update && apt-get install -y git && git clone https://github.com/ssamkyu01/Thin-Plate-Spline-Motion-Model.git && pip install -r Thin-Plate-Spline-Motion-Model/requirements.txt
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html