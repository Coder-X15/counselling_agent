# build ref: https://github.com/googleapis/python-genai
# reference for prompting system prompting: https://github.com/googleapis/python-genai#system-instructions-and-other-configs


from google import genai
from google.genai import types
from .mcp_client import *
import dotenv
import pandas as pd
import os
from datetime import datetime, timezone
import pickle


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
        
        # a mock function call to show that a request for running some soothing music has been received
        # the function has to receive the name of the song/ audio file in string
        
        self.soothing_music_tool = types.FunctionDeclaration(
                name = 'playSong',
                description = "Play a soothing song",
                parameters=None
            )
        
        # the actual tool
        self.experience_tool = types.Tool(function_declarations= [self.experience_tool])
        self.soothing_music_tool = types.Tool(function_declarations= [self.soothing_music_tool])
        

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
        
        # whole logic put in a try-except block to prevent running into errors upon interpreting the intent wrongly
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
                results = fetch_snippets(args)
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
                    condition(s). Use shorter dialogues and ask questions when necessary. If you ever feeling like turning
                    on the stereo for some soothing music will help or if the user requests it directly, suggest that you'll play some music for them and
                    play a song from the available song by calling the "playSong()" function parallely. Remember, the song is to soothe the person, so only ask them if 
                    it is helping them and not anything more.
                ''',
                tools = [self.soothing_music_tool]
            )
        )

        # append this response to the chat history

        if response.function_calls is not None:
            ## function_call_part = response.function_calls[0]
            ## self.chat_history.append(function_call_part)

            # construct a function response part
            ## result = {'results': True} # a mock result to show that the function was called
            ## function_response_part = types.Part.from_function_response(
            ##    name=function_call_part.name,
            ##    response=result
            ##)

            ## function_response_content = types.Content(
            ##    role='tool', parts=[function_response_part]
            ##)

            ## self.chat_history.append(function_response_content)

            # we let the music player widget appear
            # simulate a response to avoid errors
            mock_response = types.Content(
                role='tool',
                parts=[types.Part.from_text(text="Playing soothing music...")]
            )
            self.chat_history.append(mock_response)
            print("Setting the music player to visible")
            set_state(True)

        else:
            self.chat_history.append(
                    types.Content(
                        role = 'assistant',
                        parts = [types.Part.from_text(text = response.candidates[0].content.parts[0].text)]
                    )
                )

        ## music_prompt = self.client.models.generate_content(
        ##    model=self.model,
        ##    contents = [self.chat_history[-1]],
        ##    config = types.GenerateContentConfig(
        ##    tools = [self.soothing_music_tool]
        ##    )
        ## )
        
        ## if music_prompt.function_calls is None:
        ##    pass
        ## else:
        ##    function_call_part = music_prompt.function_calls[0]
        ##    args =  function_call_part.args['song_description']
        ##    print(f"The user might love to listen to the song:{args}")
        if response.text is not None:
            return response.text
        else:
            return self.handleNone()
        
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

    def run_once(self, query: str):
        # run the system once
        intent = self.getIntent(query)
        if 'None' in intent:
            return {'assistant': self.handleNone()}
        elif 'Fallback' in intent:
            return {'assistant': self.handleFallback()}
        elif 'Help' in intent:
            self.handleHelp()
            return {'assistant':self.handleNone()}

model = AIAgent() # resource for the app to access
if __name__ == "__main__":
    # create an instance of the agent
    agent = AIAgent()
    # run the agent
    agent.run()