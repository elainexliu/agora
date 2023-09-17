import requests
import json
import os

pos = ["happy", "calm", "neutral"]
neut = ["neutral", "surprised"]
neg = ["angry", "disgust", "fearful", "sad"]

API_URL = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
headers = {"Authorization": "Bearer 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"}

# data_folder = "../data"
data_folder = "RawData/Actor_01/"

### GETTING ALL THE INPUTS ###
inputs = []

for audio in os.listdir(data_folder):
    inputs.append(audio)

# inputs = ["03-01-01-01-01-01-01.wav", "03-01-01-01-01-02-01.wav"]

for i in range(len(inputs)):
    inputs[i] = data_folder + inputs[i]

### RESETTING OUTPUTS FOLDER ###
if os.path.exists("output"):
    for filename in os.listdir("output"):
        file_path = os.path.join("output", filename)
        os.remove(file_path)
    os.rmdir("output")
os.mkdir('output')

### GETTING OUTPUTS READ ###
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output_list = []
emotion_scores = {}

# making emotion scores for each input
for filename in inputs:
    pos_score = 0
    neut_score = 0
    neg_score = 0

    # getting the output
    output = query(filename)

    for emotion in output:
        emote = emotion["label"]
        score = emotion["score"]
        if emote in pos:
            pos_score += score
        if emote in neut:
            neut_score += score
        if emote in neg:
            neg_score += score

    emotion_scores = [pos_score, neut_score, neg_score]

    json_ret_dict = {
        "option": filename,
        "emotion_scores": emotion_scores,
        "scores": output
    }
        
    output_list.append(json_ret_dict)

# sorting the output in order of greatest to smallest pos_score
new_output_list = sorted(output_list, key=lambda d: d['emotion_scores'][0], reverse=True) 

# dumping it into a json file
with open("output/output.json", "w") as final:
   json.dump(new_output_list, final)



"""
for each of the inputs/audio files:
    make a pos / neg / neutral score
    go through each of the dictionaries, and add to the pos/neg/neutral score based on what it is

rank the audio files based on best pos score

make a json file/make the directory
in the json file:
- make a dictionary for each audio
    {"option": filename,
    "emotion_scores": [pos/neut/neg scores]
    "scores": [list of emotion dictionaries]}
    
"""