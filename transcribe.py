""" Run an alternative to AWS Transcribe """
#  Setup.sh - Setup Python
#  #!/bin/sh
#  sudo dnf install python3.11 -y
#  sudo dnf install python3.11-pip -y
#  python3.11 -m venv test_env
#  . ./test_env/bin/activate
#  pip3.11 install pyannote.audio
#  pip3.11 install openai-whisper
#  pip3.11 install torch
#  pip3.11 install pytorch::torchaudio
#  ## Setup
#  `conda create -n transcribe jupyter `
#  `conda activate transcribe`
#  `pip install pyannote.audio`
#  `pip install openai-whisper`
#  `conda install pytorch::torchaudio`
#  also install BenderLibIsengard to make it run the Model, too
#  # Copy files from HuggingFace for the models and update the model YAML and
# the model names.  I have a ZIP of the entire project.

import os
import json
from pathlib import Path
import warnings
from argparse import ArgumentParser
import signal
#import time
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import whisper
from tqdm import tqdm

import torch
#import torchaudio

import boto3



def timeout_handler(num, stack):
    """ Set a SigALRM Processor for any stuck segments in processing """
    print("Received SIGALRM, exiting...")
    raise Exception("Timeout")

signal.signal(signal.SIGALRM, timeout_handler)

def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    """ This builds the Pyannote Pipeline from the pretrained model. """
    
    path_to_config = Path(path_to_config)

    #print(f"Loading pyannote pipeline from {path_to_config}...")
    # the paths in the config are relative to the current working directory
    # so we need to change the working directory to the model path
    # and then change it back

    cwd = Path.cwd().resolve()  # store current working directory
    #print (f"Current working directory: {cwd}")

    # first .parent is the folder of the config, second .parent is the folder containing the
    # 'models' folder
    cd_to = cwd.parent.resolve()

    #print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    #print(f"Loading Pipeline from {cd_to}/{path_to_config}")
    # made this a try/catch to avoid losing the CWD path in the notebook
    try:
        pipeline = Pipeline.from_pretrained(path_to_config)
        os.chdir(cwd)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        os.chdir(cwd) # force CWD restore
        raise e
    #print(f"Changing working directory back to {cwd}")

    return pipeline

prompt="""Please analyze and summarize the following video transcript. In your summary, include:
1. Main speakers and their roles/perspectives
2. Key topics discussed
3. Key statistics and metrics covered
4. Major themes that emerge throughout the conversation
5. Important points made by each speaker
6. Any notable quotes or standout moments
7. The overall tone and atmosphere of the discussion
8. Any conclusions or calls to action presented
9. A brief overall summary in 2 to 4 sentences that captures the essence of the trancript content.

For your output, please follow the directions:
* Begin you analysis with the phrase "Transcript Summary:" for items 1-8 above, and "Overall Summary:" for the final summary
* Structure your summary with clear headings for each of these elements. 
* Use bullet points where appropriate to highlight specific details and timecode of outlined topic. 
* If the transcript includes timestamps, reference any particularly significant moments by their timestamp.
* Provide metadata tags based on a News ontology
* Note any areas of agreement or disagreement among speakers
* Highlight any questions that were left unanswered or topics that seem to require further discussion
"""

warnings.filterwarnings("ignore")

""" Get the parameters """
parser = ArgumentParser(
    prog='transcribe.py',
    description='Run pyannote diarization and whisper transcription on an audio file',
    epilog='this runs as one single-threaded monster.  ' +
            'Use transcriballel.sh to parallel process on non-GPU system'
    )
parser.add_argument("-i", "--input", dest="INPUTAUDIO",
                    help="Input Audio File (path to) to process.",
                    metavar="FILE",
                    required=True
                    )
parser.add_argument("-o", "--output", dest="OUTPUTFILE",
                    help="Output File (path to) to write. Defaults to InputFile.transcript.json",
                    metavar="FILE", required=False
                    )
parser.add_argument("-s", "--summary", dest="SUMMARYFILE",
                    help="Output File (path to) to write. Defaults to InputFile.transcript.json",
                    metavar="FILE", required=False
                    )
parser.add_argument("-p", "--prompt", dest="PROMPTFILE",
                    default="data/news_prompt.txt",
                    help="Prompt File (path to) to use for summarization. Defaults to prompt.txt",
                    metavar="FILE", required=False
                    )
parser.add_argument("-t", "--timeout", dest="ALARM_TIMEOUT",
                    default=60,
                    help="Timeout in seconds for whisper to process each segment. Defaults to 60",
                    metavar="INT",
                    required=False
                    )
args = parser.parse_args()

INPUTAUDIO=""
PROMPTFILE=''
SUMMARYFILE=""


if args.INPUTAUDIO:
    INPUTAUDIO=args.INPUTAUDIO
if args.OUTPUTFILE:
    OUTPUTFILE=args.OUTPUTFILE
else:
    OUTPUTFILE=INPUTAUDIO+".transcript.json"
if args.ALARM_TIMEOUT:
    ALARM_TIMEOUT=int(args.ALARM_TIMEOUT)
else:
    ALARM_TIMEOUT=60
if args.PROMPTFILE:
    PROMPTFILE=args.PROMPTFILE
if args.SUMMARYFILE:
    SUMMARYFILE=args.SUMMARYFILE




PATH_TO_CONFIG = "models/pyannote_diarization_config.yaml"

#pipeline= Pipeline.from_pretrained(PATH_TO_CONFIG)
pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)
# send pipeline to GPU when available
if ENABLE_CUDA:
    print("Enabling CUDA on Torch.  Pray for your soul.")
    pipeline.to(torch.device("cuda"))
else:
    print("No CUDA available.  This will run slowly (10x realtime)")
print("Pipeline Loaded Successfully")


# ## Step 1 - run through pyAnnote.audio to parse identify markers of different speakers
#
# Offline running tutorial is found
# #[here](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb) # pylint: disable=c0301
#

#load to memory
#waveform, sample_rate = torchaudio.load(INPUTAUDIO)
#diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

# or just run on the file itself
with ProgressHook() as hook:
    diarization = pipeline(INPUTAUDIO, hook=hook)


# Save all the segments by Speaker, in a format consistent with AWS transcribe
# save the result
segments = {}
sequenceNumber = 0
for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
    #print(speech_turn)
    if speaker not in segments:
        segments[speaker] = []
    segments[speaker].append( {
                "id": sequenceNumber,
                "start_time": speech_turn.start, 
                "end_time": speech_turn.end,
                "speaker_label": speaker,
                "transcript" : ""
    } )
    sequenceNumber += 1
    #print(f"start={speech_turn.start} stop={speech_turn.end} speaker={speaker}")

print(f"Mapped {sequenceNumber} segments by Speaker")

model = whisper.load_model("turbo")

# Function to do inference on a segment so we can alarm.
def run_transcribe(amodel, audio_file, timestamp_list):
    """
    Runs Transcribe itself on a chunk of the audio file.
    """
    result = amodel.transcribe(
                audio=audio_file,
                clip_timestamps= timestamp_list
    )
    return result





output = [ ]
PROCESS_LIST = [] # Make a list of IDs if you are having trouble and onlly want to process individual segments
#PROCESS_LIST = [504, 505, 506, 507]
processed=0
#for speaker in segments: # pylint: disable=C0206
for item in tqdm(segments):
    #print(f"Processing Speaker: {speaker}")
    #for item in segments[speaker]:

    while item["id"] >= len(output):
        #print(f"Extending Output (current length={len(output)}) "
        #       f"because this is sequence {item["sequence"]+1}")
        output.append( {} )

    signal.alarm(ALARM_TIMEOUT)
    try:
        #if len(PROCESS_LIST) >0 and item["id"] not in PROCESS_LIST:
        #    print(f"Skipping Item # {item['id']}")
        #    result = {"text": "[unknown]"}
        #else:
        result = run_transcribe(amodel=model,
            audio_file=INPUTAUDIO,
            timestamp_list=','.join([str(item["start_time"]),str(item["end_time"])]))
    except Exception as ex:
        print("Exception: " + str(ex))
        result = {"text": "[TRANSCRIPTION SKIPPED]"}
        # re-load the model
        model = whisper.load_model("turbo")
    finally:
        signal.alarm(0)

    output[item["id"]] = {
            "id": item["id"],
            "speaker_label": speaker,
            "start_time": item["start_time"],
            "end_time" : item["end_time"],
            "transcript": result["text"]
    }
#        processed += 1
#        percentage = float(processed)/float(sequenceNumber)
#        print(f"processed {processed}/{sequenceNumber} ( { percentage:.1%} ) item # {item['id']} : {speaker} - {item['start_time']} to {item['end_time']}")


#print(json.dumps(output, indent=4))
# output the contents of output to a file OUTPUT_FILE

with open(OUTPUTFILE, 'w') as f:
    f.write(json.dumps(output, indent=4))


# Now call Bedrock Claude 3 Haiku with the prompt from PROMPT and including the output variable as additional data for the model
# This will be the final summary of the transcript
prompt = open(PROMPTFILE, encoding='utf-8').read()

client = boto3.client('bedrock-runtime')
modelId = 'anthropic.claude-3-haiku-20240307-v1:0'
accept = 'application/json'
contentType = 'application/json'

native_request = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 2048,
    "temperature": 0.1,
    "messages": [{"role": "user", 
                  "content": [ {
                      "type": "text", 
                      "text": prompt + "\n" + json.dumps(output)
                  } ]
    }]
}
request = json.dumps(native_request)


response = client.invoke_model(
    modelId=modelId,
    body=request
)
response_body = json.loads(response.get('body').read())
with open(SUMMARYFILE, 'w') as f:
    f.write(response_body.get('content')[0].get('text'))
    print(f"Summary saved to {SUMMARYFILE}")
