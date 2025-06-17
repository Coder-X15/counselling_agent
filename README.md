## Counselling Agent
Capabilities:
1. RAG using Pandas on a dataset from [Amod's mental helath counselling conversations dataset](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)
2. Google Gemini 2.0 Flash model in use.
3. Use of keyword extraction/inference from the conversation to ease advice development.

## How to Use:
### 1. Install Prerequisites
- In the repo folder, run `pip install -r requirements.txt` to install the required packages. (preferably in a virtual environment if you'd like to keep the dependencies isolated)
- **For the developers**: if you'd like to run tests, do `pip install -r requirements-dev.txt`.
### 2. Running the Agent:
There are two ways:
1. Run `python ./agent/global_agent.py` in the root (or run that file within its parent folder; either will work)
2. Create a new Python file in the root, add the following:
```
from agent import *

if __name__ == "__main__":
    agent = AIAgent()
    agent.run()
```

## Possibly Unwanted Pieces of Code:
1. Function `write_cache` in `./agent/global_agent.py`: I don't really think we'll need to cache formerly accessed dataset rows, it's extra memory consumption in the cloud, if ever done so.

## Todo:
- Add MCP interface to play soothing music if required in the chat interface.