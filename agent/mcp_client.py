import requests

def fetch_snippets(keywords: list) -> str:
    response = requests.post("https://counsellor-mcp-server.vercel.app/tools/fetchConversationSnippets"
                             , json={"arguments": {'kwds': keywords}}
                             )
    if response.status_code == 200:
        return response.json()[0]['text']
    
def set_state(state : bool) -> bool:
    response = requests.post(
        "https://counsellor-mcp-server.vercel.app/tools/setState",
        json={"arguments": {"state": state}}  # replace with actual state value
    )
    if response.status_code == 200:
        return response.json()[0]['text'] == "true"
    
def get_state() -> bool:
    response = requests.post(
        "https://counsellor-mcp-server.vercel.app/tools/getState",
        json={"arguments": None}
    )
    if response.status_code == 200:
        return response.json()[0]['text'] == "true"
