import requests

def fetch_snippets(keywords: list) -> str:
    response = requests.post("https://counsellor-mcp-server.vercel.app/tools/fetchConversationSnippets"
                             , json={"arguments": {'kwds': keywords}}
                             )
    if response.status_code == 200:
        return response.json()[0]['text']

