from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.0, max_tokens=1000)

search_tool = TavilySearchResults()

@tool
def get_system_time(format:str= "%Y-%m-%d %H:%M:%S") -> str:
    """Get the current system time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [search_tool, get_system_time]

agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")