# build ref: https://github.com/googleapis/python-genai
# reference for prompting system prompting: https://github.com/googleapis/python-genai#system-instructions-and-other-configs


from google import genai
from google.genai import types
import dotenv
import pandas as pd
import os
from datetime import datetime, timezone
import pickle

# loading resources
mental_health_df = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)

# We'll check if cache of
if not os.path.exists('./cache'):
    os.mkdir('./cache')

# function to write cache from an array containing indices of frequently accessed records
def write_cache(indices: list):
    # create the filename using the current date in the current timezone
    # CHECKS:
    if any([type(k) != int for k in indices]):
        raise TypeError("Integers only should be used as indices}.")
    elif indices == []:
        raise ValueError("List of indices cannot be empty.")
    
    filename = f'cache/{datetime.now(timezone.utc).strftime("%Y%m%d")}.cache'
    with open(filename, 'ab') as f:
        pickle.dump(mental_health_df.iloc[indices], f)

# search dialogues with keywords
# logic : check for the keywords in both columns ('Context' and 'Response' columns) and get
# the indices of the matching conversations
def search_conversations(keywords: list):
    # create a list to store the indices of matching conversations
    # CHECKS:
    if any([k == '' for k in keywords]):
        raise ValueError("Empty keywords are not allowed.")
    elif any([type(k) != str for k in keywords]):
        raise TypeError("Keywords must be strings.")
    
    indices = []
    # iterate over each keyword
    for keyword in keywords:
        # iterate over each row in the dataframe
        for index, row in mental_health_df.iterrows():
            # check if the keyword is present in the 'Context' or 'Response' column
            if keyword in row['Context'] or keyword in row['Response']:
                # if the keyword is found, add the index to the list
                indices.append(index)
                # remove duplicates from the list of indices
                indices = list(set(indices))
    # return the relevant conversational items to the user as a sub-dataframe
    return mental_health_df.iloc[indices]


# laoding environement configs
dotenv.load_dotenv()

# Agent Development Plan
# 1. The agent must be able to identify the intent from the user's input.
# 2. The agent must deny homework, coding and related help but instead focus 
#    on helping the user cope up with their mental health issues.
# 3. The agent must be able to provide the user with resources and information 
#    about mental health, and guide them through possible remedies, if any.

# References for the AI agent to find remedies
# 1. NIMH articles
# 2. Psychology Today articles
# 3. Mental Health America articles
# 4. Mayo Clinic articles

# Startegies to take upon suicial tendencies while in chat:
# 1. Redirect the user to a crisis hotline
# 2. Use a mail API to email the admin regarding the same (ig)
class AIAgent:

    def __init__(self):
        # loading the chat client

        # Gemini chat client
        self.client = genai.Client(
            api_key=os.environ['GOOGLE_API_KEY'],
            http_options=types.HttpOptions(api_version='v1alpha')
        )

        # model to be used
        self.model = 'gemini-2.0-flash-001'

        # System prompt for intent detection
        # Intended agent behaviour
        # 1. The agent must be able to identify user intents
        # 2. The agent must reject homework help or STEM problems or questions
        # 3. The agent must reject programming questions
        self.intent_prompt = '''
        Identify the user intent from the input provided.
        ## None intent:
        Return "Intent:None" for a user input similar to a simple dialogue or a casual question.
        ## Help Trigger Intent:
        Return "Intent:Help" for inputs that are related to the user's mental health or current situation for which they seek advice.
        ## Fallback Intent:
        Return "Intent:Fallback" for STEM doubts, homework and programming help.
        '''

        # chat history
        self.chat_history = []

        # number of the last replies to be taken into the context window
        self.context_window_size = 10

        # a tool to get conversation snips for developing advice regarding the person's mental health
        self.experience_tool = types.FunctionDeclaration(
                name = 'fetchAdvice',
                description = "Infer keywords from the chat that indicates the user's feelings, their current situations, and the like, return them as a Python list",
                parameters = types.Schema(
                        type = 'OBJECT',
                        properties = {
                            'keywords': types.Schema(
                                type = 'ARRAY',
                                items = types.Schema(
                                    type = 'STRING',
                                    description= 'A keyword to be used for developing advice.',
                                ),
                                description = 'A list of keywords to be used for developing advice.',
                            )
                        },
                        required = ['keywords']
                    ),
            )
        
        # the actual tool
        self.experience_tool = types.Tool(function_declarations= [self.experience_tool])

    def getIntent(self, user_input : str):
        # get the intent from the user input
        self.chat_history.append(
            types.Content(
                role = 'user',
                parts = [types.Part.from_text(text = user_input)]
            )
        )

        # send the user input to the chat client
        response = self.client.models.generate_content(
            model=self.model,
            contents = user_input,
            config = types.GenerateContentConfig(
                system_instruction = self.intent_prompt,
            )
        )

        return response.text
    
    def handleFallback(self):
        # handle the fallback intent
        response = self.client.models.generate_content(
            model=self.model,
            contents = self.chat_history,
            config = types.GenerateContentConfig(
                system_instruction = '''
                    Reply to the user with an apology that you are unable to assist with their request.
                ''',
            )
        )

        return response.text
    
    def handleHelp(self):
        # handle the help intent
        # Howto:
        # 1. Separate out the keywords from the user input
        # 2. Use the keywords to search out relevant chat instances and add them back in
        
        try:
            # step 1
            response = self.client.models.generate_content(
                model=self.model,
                contents = self.chat_history,
                config = types.GenerateContentConfig(
                    tools = [self.experience_tool]
                )
            )
            function_call_part = response.function_calls[0]
            function_call_content = response.candidates[0].content
            args = results = None

            # step 2
            try:
                # get the arguments to the function
                args = function_call_part.args['keywords']
                results = search_conversations(args).to_dict().values()
                results = list(results)[1]
                results = list(results.values())
                results = " ".join(results)
                results =  self.client.models.generate_content(
                model=self.model,
                contents = [types.Part.from_text(text = results)],
                config = types.GenerateContentConfig(
                    system_instruction= '''Summarize the conversation snips.'''
                )
            )
                results = {'results': results}
            except:
                raise Exception("Unable to advise you, pal.")
            
            # append the different parts generated to the chat history
        
            function_response_part = types.Part.from_function_response(
                name=function_call_part.name,
                response= results,
            )

            function_response_content = types.Content(
                role='tool', parts=[function_response_part]
            )

            self.chat_history.append(function_call_content)
            self.chat_history.append(function_response_content)
        except:
            pass

    def handleNone(self):
        # handle the None intent
        response = self.client.models.generate_content(
            model=self.model,
            contents = self.chat_history,
            config = types.GenerateContentConfig(
                system_instruction = '''
                    You are a helpful AI counsellor bot. Help the user cope up with their mental health issues.
                    Be empathetic and non-judgmental, forgive them their tirades using abusive words and phrases.

                    In the light of the chat history, reply to the user appropriately. Speak more like a human counselor,
                    do not be too formal or robotic. Use your best judgment to decide what to say. Help the user analyze
                    their situation but do not do so like a robot. Ask questions, get more clarity regarding their
                    condition(s). Use shorter dialogues and ask questions when necessary.
                ''',
            )
        )

        # append this response to the chat history
        self.chat_history.append(
            types.Content(
                role = 'assistant',
                parts = [types.Part.from_text(text = response.text)]
            )
        )
        
        return response.text

    def run(self):
        # run the system
        # flow:
        # 1. Analyze intent
        # 2. If None intent found, handle like a regular conversation
        # 3. If fetch recipe intent found, call the handleFetchRecipe function, add the recipe to the system chat memory
        # 4. Use the chat memory as the context for the system to keep the conversation going
        while True:
            user_input = input("User:")
            intent = self.getIntent(user_input)
            if 'None' in intent:
                print(f"Assistant: {self.handleNone()}")
            elif 'Fallback' in intent:
                print(f"Assistant: {self.handleFallback()}")
            elif 'Help' in intent:
                self.handleHelp()
                print(f"Assistant: {self.handleNone()}")

if __name__ == "__main__":
    # create an instance of the agent
    agent = AIAgent()
    # run the agent
    agent.run()