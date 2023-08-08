import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
import streamlit as st
from langchain.schema import SystemMessage
from fastapi import FastAPI

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY ")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text

# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content ased on objective if the content 
    #objective is the original objective & task that user gives to agent, url is the 

    print("Scraping website...")
    # define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
    }

    # define the data to be sent in the request
    data = {
        "url": url
    }

    # convert python object to JSON string
    data_json = json.dumps(data)

    # send the POST request
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTT", text)

        if len(text) >10000:
            output = summary(objective,text)
            return output
        else: 
            return text
    else: 
        print(f"HTTP request failed with status code {response.status_code}")

def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}: 
    "{text}"
    Summary:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])
    
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output

class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(description="The objective & task that user gives to agent")
    url: str = Field(description="The url of the website to be scraped")

class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any data or urls"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)
    
    def _arun(self, url: str):
        raise NotImplementedError("error here")

# 3. Create langchain agent with tools above

tools = [
    Tool(
        name = "Search",
        func = search,
        description = "useful for when you need to answer a question about current events or look up supporting information to give a better answer. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a word class researcher, who can do detailed research on any topic and produce facts based results using your available tools;
    You do not make things up, you will try as hard as possible to gather facts & data to support your research
    
    Please make sure you complete the objective above with the following rules:
    1/ You should do enough research to gather as much information as possible about the objective
    2/ If there are url of relevant links & articles, you will scrape it to gather more information
    3/ After scraping & searching, you should think "is there any new things I should research & scrape based off of the data i have found to improve my research quality?" If anser is yes, continue; But don't do this more than 3 iterations
    4/ You should not make things up, you should only write fats & data that you have gathered
    5/ In the final output, you should include all refrence data & links to back up your research; you should include all relevant links to back up your research
    6/ In the final output, you should include all refrence data & links to back up your research; you should include all relevant links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model = "gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs = agent_kwargs,
    memory = memory,
)


''' # use streamlit to create a web app
def main():
    st.set_page_config(page_title="WS Ai Research Agent", page_icon=":bird:")

    st.header("WeedSociety's Ai Research Agent :bird:")
    query = st.text_input("Research Topic..")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main() '''

# 5. Set up webservice API

app = FastAPI()

class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return content