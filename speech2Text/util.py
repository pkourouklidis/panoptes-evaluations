import requests
import torch
from essentia.standard import MonoLoader, NoiseAdder
import pandas as pd

_, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_stt', language="en")

def sendInfernceRequest(data):
    body = {
            "id": "1",
            "inputs": [
                {
                    "name": "input",
                    "shape": [1,len(data)],
                    "datatype": "FP32",
                   "data": [data]
                }
            ],
            "outputs": [{"name": "output"}]
        }
    response = requests.post(
            "http://stt-predictor-default.evaluations.panoptes.uk/v2/models/stt/infer",
            json = body,
            headers = {"accept-encoding": None}
        )
    shp = response.json()["outputs"][0]["shape"]
    output = response.json()["outputs"][0]["data"]
    output = torch.Tensor(output).reshape(shp)
    return decoder(output[0])

def readFile(path):
    loader = MonoLoader(filename=path, sampleRate = 16000)
    audio = loader()
    return audio.tolist()

def loadToNumpy(path):
    loader = MonoLoader(filename=path, sampleRate = 16000)
    audio = loader()
    return audio

def generateRows(start, end):
    validated = pd.read_csv("cv-corpus-12.0-delta-2022-12-07/en/validated.tsv", sep="\t")
    for row in validated[start:end].itertuples():
        yield row._replace(path = "cv-corpus-12.0-delta-2022-12-07/en/clips/" + row.path) 
        
def addNoise(wav):
    noiseAdder = NoiseAdder(level=-20)
    noisyAudio = noiseAdder(wav)
    return noisyAudio