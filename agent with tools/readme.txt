this is an example of a tool using agent using langchain library

agent is using:
	python
	duckduckgo for web search
	bs4 for web scrape
	text RAG and chromaDB for local search
	get current time
	save file
as its tools
	

the agent is built on groq supported LLMs (though it can be easily adapted to local or big cloud LLM uses)
to use the agent apart of python and the libraries (langchain, regex, duckduckgo): 
	1. the groq API is required.
	2. the matching chromedriver is required (for web scraping)
