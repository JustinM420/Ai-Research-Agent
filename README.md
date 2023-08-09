# Ai-Research-Agent

## Project Overview

The WS Ai Research Agent is designed to simulate a world-class researcher. This agent is capable of conducting detailed research on any topic and producing factual results using the tools available to it. It achieves this by leveraging web scraping, search functionality, and AI-driven summarization.

## Features

1. **Search Tool**: Searches for a given query and returns relevant results.
2. **Web Scraping Tool**: Scrapes content from a given URL and, if the content is lengthy, summarizes it based on the given objective.
3. **AI-Driven Summarization**: Uses the GPT-3.5 model to produce concise summaries of long texts.
4. **Agent Initialization**: Configures the agent with the aforementioned tools and rules to ensure factual and reliable results.
5. **Streamlit GUI**: User can interact with agent via Streamlit GUI or comment it out and utilize the Fast API code to create a web service.
6. **FastAPI Integration**: Provides an API endpoint for users to post their research queries and get researched content in return, creating a web service that can be integrated with zapier or zapier like tools to create a workflow.

## Setup & Installation

### Prerequisites:

1. Python 3.x
2. Required environment variables: `BROWSERLESS_API_KEY`, `OPENAI_API_KEY`, `SERP_API_KEY`.

### Installation Steps:

1. Clone the repository.
```bash
git clone <repository_url>
```
2. Navigate to the project directory.
```bash
cd path/to/project
```
3. Install the required packages.
```bash
pip install -r requirements.txt
```
(Note: You may want to create a virtual environment before installing the packages.)

4. Create a `.env` file in the root directory and add the following content:
```
BROWSERLESS_API_KEY=<Your_Browserless_API_Key>
SERP_API_KEY=<Your_SERP_API_Key>
```
Replace `<Your_Browserless_API_Key>` and `<Your_SERP_API_Key>` with your actual API keys.

5. Run the FastAPI application.
```bash
uvicorn <filename>:app --reload
```
Replace `<filename>` with the name of the Python file containing the FastAPI app.

## Usage:

To use the WS Ai Research Agent API:

1. Start the FastAPI server (as described above).
2. Send a POST request to the root endpoint (`/`) with the research query in the body. For example:
```json
{
    "query": "Your research topic here"
}
```
3. The API will respond with the researched content.

## Rules for the Agent:

1. Do enough research to gather as much information as possible about the objective.
2. If there are URLs of relevant links & articles, scrape them to gather more information.
3. After scraping & searching, think: "is there anything new I should research & scrape based on the data I've found to improve my research quality?" If the answer is yes, continue. However, don't perform more than 3 iterations.
4. Only present facts & data gathered. Do not make things up.
5. Include all reference data & links in the final output to back up your research.

## Contributing:

If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Licensing:

The code in this project is licensed under MIT license.

---

*Note: This README is a guideline. You might want to modify it according to the specifics of your project and add any additional sections if required.*
