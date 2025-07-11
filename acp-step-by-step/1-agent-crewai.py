# ############################################################# 
#
# This example makes use of local ollama to serve the models
#
# 1. Make sure to have ollama locally installed and running
# 2. Download target LLM model from https://ollama.com/library
# 3. Verify with ollama ls
#
# ############################################################# 
#
# A simple examplke to create an agent with the specified model
# 
# python3 ./1-agent-crewai.py -m ollama/llama3:8b
# python3 ./1-agent-crewai.py -m ollama/qwen2.5:14b
#
# ############################################################# 

import argparse

from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

def cli_parse():
    parser = argparse.ArgumentParser(description='Run LLM examples with different LLMs and provider API Keys.')
    parser.add_argument('-m', '--model', type=str, help='source/model name')
    parser.add_argument('-u', '--url', type=str, help='LLM URL of the')

    args = parser.parse_args()
    if args.model is None:
        print ("\nSpecify a model name")
        exit(0)
    if args.url is None:
        url = "http://localhost:11434"
    else:
        url = "http://"+args.url

    return args.model, url

def get_llm(llm_model, llm_url):
    return LLM(model=llm_model, base_url=llm_url, max_tokens=8192)

def get_researcher_agent(llm):
    return Agent(
        role='Senior Researcher',
        goal='Be the best researcher in the world',
        backstory="You are a Senior Researcher at a leading tech company. Your job is to research and find the best solutions to complex problems.",
        llm=llm,
        verbose=True
    )

def get_researcher_task(researcher):
    return  Task(
        description='Research the latest advancements in AI',
        agent=researcher,
        expected_output='A comprehensive report on the latest AI advancements'
    )

# Get the model and url
llm_model, llm_url = cli_parse()

# Create LLM object
llm = get_llm(llm_model, llm_url)

# Create researcher agent
researcher_agent = get_researcher_agent(llm)

# Create researcher task
researcher_task = get_researcher_task(researcher_agent)

# Create crew object
crew = Crew(agents=[researcher_agent], tasks=[researcher_task], verbose=True)

# kickoff task
task_output = crew.kickoff()

# print output
print(task_output)

