import os
import regex as re
import docx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from duckduckgo_search import DDGS

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Defining tools
@tool
def get_current_time(*args, **kwargs):
    """
    return the current time in d-month-y H:MM format.
    Use ONLY if: asked for a current time OR you need
    to know a current day / time for other operations
    """
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%d-%b-%Y %H:%M:%S")
@tool
def get_surgical_info(prompt: str) -> str:
    """
    function to search in a local database

        Args: prompt (str): the user question to pass to the similarity-search in the local database

    Returns:
        str: The context from the vector store about the question.

    useful if the question contains background of a surgical operations
    """
    from langchain_chroma import Chroma
    from langchain_nomic.embeddings import NomicEmbeddings
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    print(f"\n\n========                       searching chroma_db database                        ========")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    results = db.similarity_search_with_score(prompt, k=1)
    for doc, _score in results:
        print(f"Score: {_score}")
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text
@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return the result.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: The output of the code execution.
    """
    try:
        local_ns = {}

        # Execute the code
        exec_result = exec(code, {}, local_ns)

        # Return the variables defined in the local namespace
        result = str(local_ns)

        # If there was an actual result from exec, use that too
        if exec_result is not None:
            result += f"\nExec result: {exec_result}"

        return result
    except Exception as e:
        return f"Error: {str(e)}"
@tool
def simple_web_search(query: str) -> str:
    """
    Search the web for information using DuckDuckGo ONLY if the url is not given!.

    Args:
        query (str): The search query.

    Returns:
        str: The search results, links.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            href = result.get('href', 'No link')
            formatted_results.append(f"{i}. {title}\n{body}\nSource: {href}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {str(e)}"
def web_scrape(prompt):
    import selenium.webdriver as webdriver
    from selenium.webdriver.chrome.service import Service
    from bs4 import BeautifulSoup
    import time

    url = re.findall(r'https?://\S+', prompt)[0]
    print(f"URL: {url}")
    prompt = "{in this url}".join(prompt.split(url))
    print(f"Prompt: {prompt}")

    def scrape_website(website):
        print("Launching browser...")

        chrome_driver_path = "./chromedriver.exe"
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

        try:
            driver.get(website)
            print("Page loaded...")
            html = driver.page_source
            time.sleep(5)

            return html
        finally:
            driver.quit()
    def extract_body_content(html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        body_content = soup.body
        if body_content:
            return str(body_content)
        return ""
    def clean_body_content(body_content):
        soup = BeautifulSoup(body_content, "html.parser")

        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        cleaned_content = soup.get_text(separator="\n")
        cleaned_content = "\n".join(
            line.strip() for line in cleaned_content.splitlines() if line.strip()
        )

        return cleaned_content
    def split_dom_content(dom_content, max_length=6000):
        return [dom_content[i:i + max_length] for i in range(0, len(dom_content), max_length)]
    def parse_with_ollama(dom_chunks, user_prompt):
        model = ChatGroq(model="llama3-70b-8192")

        template = ("<role>You are tasked with extracting specific information from the following text content: {dom_content}.</role>"
            "<instruction>Please follow these instructions carefully: </instruction>"
            "<instruction>Only extract the information that directly matches the provided description: {user_prompt}. </instruction>"
            "<instruction>Do not include any additional text, comments, or explanations in your response. </instruction>"
            "<instruction>If no information matches the description, return an empty string ('').</instruction>"
            "<instruction>Your output should contain only the data that is explicitly requested, with no other text.</instruction>")

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        parsed_results = []

        for i, chunk in enumerate(dom_chunks, start=1):
            response = chain.invoke(
                {"dom_content": chunk, "user_prompt": user_prompt}
            )
            print(f"Parsed batch: {i} of {len(dom_chunks)}")
            parsed_results.append(response.content)
        first_level_result = "\n\n".join(parsed_results)

        template_2 = (
            "<instruction>given the {dom_content} rephrase it into a nice, organised text, use markdown format if text is large and structured.</instruction>"
            "<instruction>give the answer in flowing text. avoid redundancies and include the most important information.</instruction>")
        prompt = ChatPromptTemplate.from_template(template_2)
        chain = prompt | model
        second_level_result = chain.invoke({"dom_content": first_level_result}).content

        return [first_level_result, second_level_result]

    result = scrape_website(url)

    body_content = extract_body_content(result)
    cleaned_content = clean_body_content(body_content)
    #print(cleaned_content)
    dom_chunks = split_dom_content(cleaned_content)
    result = parse_with_ollama(dom_chunks, prompt)

    return result
@tool
def save_python_file(filename: str, content: str) -> str:
    """
    Save Python code to a file.

    Args:
        filename (str): The name of the file (should end with .py).
        content (str): The Python code to save.

    Returns:
        str: Confirmation message.
    """
    if not filename.endswith('.py'):
        filename = f"{filename}.py"

    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully saved Python code to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"
@tool
def save_document(filename: str, content: str) -> str:
    """
    Save content to a Word document (.docx).

    Args:
        filename (str): The name of the file (should end with .docx).
        content (str): The text content to save.

    Returns:
        str: Confirmation message.
    """
    if not filename.endswith('.docx'):
        filename = f"{filename}.docx"

    try:
        doc = docx.Document()

        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                doc.add_paragraph(para.strip())

        doc.save(filename)
        return f"Successfully saved document to {filename}"
    except Exception as e:
        return f"Error saving document: {str(e)}"
@tool
def save_text_file(filename: str, content: str) -> str:
    """
    Save content to a plain text file.

    Args:
        filename (str): The name of the file.
        content (str): The text content to save.

    Returns:
        str: Confirmation message.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully saved text to {filename}"
    except Exception as e:
        return f"Error saving file: {str(e)}"


tools = [
    python_repl,
    simple_web_search,
    get_surgical_info,
    get_current_time,
    save_python_file,
    save_document,
    save_text_file,
]


system_message = """<role>You are a helpful AI assistant that can chat with user, answer questions, 
perform calculations using Python, search the web for recent information, scrape page and answer from the page, and create files. Be accurate, helpful, and respond in a conversational manner.</role>
<instruction>For answer relevant to current time, use get current time function, apart of other tools (if necessary)</instruction>
<instruction>For any math problems or calculations, use the Python tool rather than calculating yourself.</instruction>
<instruction>For a general questions about recent events if URL IS NOT GIVEN use the web search tool.</instruction>
<instruction>If the question is about surgery, do not hesitate to check the local database through get_surgical_info function first!</instruction>
<instruction>If a url is given with a specific question, perform a scrape_page function and nothing else.</instruction>
<instruction>When asked to create or write files, use the appropriate tool:
- save_python_file for Python scripts (.py)
- save_document for Word documents (.docx)
- save_text_file for plain text files</instruction>
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

"""def L1_classification_agent(user_input):
    "
        Classifies a user query into one of five categories: general, sales, logistics, research, or database.

        Args:
            user_query (str): The query from the user

        Returns:
            str: The classification category (one word only)
    "
    categories = ["python", "general", "web_search", "rag"]
    model = ChatGroq(
        model="qwen-2.5-32b",
        temperature=0
    )
    sample_data = {
        'query': [
            "code a data science project with iris dataset using log. reg. model",
            "How many products did we sell last month?",
            "what are the news about Elon Musk?",
            "what are operation steps of lichtenstein",
            "Save document to a file document.docx",
            "https://en.wikipedia.org/wiki/Mary_Anne_MacLeod_Trump who did she marry and when?",
            "create a python file with a code above",
            "write a simple fibonacci code",
            "Can you find information about renewable energy?",
            "what is 3*89",
            "what is the time?"
        ],
        'category': [
            "python",
            "general",
            "web_search",
            "rag",
            "system",
            "web_search",
            "system",
            "python",
            "web_search",
            "python",
            "system"
        ]
    }
    df = pd.DataFrame(sample_data)
    prompt_template = "
    You are a query classification agent. Your only task is to classify user queries into exactly one of these categories:
    {categories}

    Here are some examples of classified queries:
    {examples}

    User query: {query}

    Respond with exactly ONE WORD from the categories list. No explanation, no reasoning, just the category name. 
    Choose the most appropriate category based on the query intent.
    "

    def format_examples(df: pd.DataFrame) -> str:
        examples = []
        for _, row in df.iterrows():
            examples.append(f"Query: \"{row['query']}\" → Category: {row['category']}")
        return "\n".join(examples)

    examples = format_examples(df)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    classification_chain = ({"query": RunnablePassthrough(),
                             "categories": lambda _: ", ".join(categories),
                             "examples": lambda _: examples}
                            | prompt
                            | model
                            | StrOutputParser())

    try:
        result = classification_chain.invoke(user_input)
        # Ensure the result is one of the valid categories
        result = result.strip().lower()
        if result in categories:
            return result
        else:
            return "general"  # Default fallback
    except Exception as e:
        print(f"Error classifying query: {e}")
        return "general"  # Default fallback"""

#   available groq_llms
groq_models = ["llama3-70b-8192", "llama-3.3-70b-versatile", "llama-3.3-70b-specdec", "qwen-2.5-32b"]

# Set up the agent
def set_agent(model, tools):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=model,
        temperature=0.5,
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    return agent_executor

"""def L3_correction_agent(user_input, response):
    model = ChatGroq(
        model="qwen-2.5-32b",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0
    )
    prompt1 = f"does the response {response} answer the user  question
    {user_input} in a good quality? 
    <instruction>answer with 'Yes' or 'No'.</instruction> 
    <instruction>Answer only with one of these two words.</instruction>
    <instruction>nothing else, no explanations</instruction>
    "
    print(model.invoke(prompt1).content.lower())
    if model.invoke(prompt1).content.lower() == "yes":
        return None
    return True"""

def main():
    chat_history = []

    print("Type 'exit' to end the conversation")


    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye! 👋")
            break

        if user_input == "-cc":
            chat_history = []
            print(" -= chat history cleared =- ")

        if "http" in user_input:
            result = web_scrape(user_input)
            #print(f"\n\nFirst level result: \n{result[0]}\n\n" )
            agent_response = result[1]
            print(f"\nAgent: \n{agent_response}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        else:
            for index, model in enumerate(groq_models):
                try:
                    agent_executor = set_agent(model, tools)

                    result = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                        "agent_scratchpad": [],
                    })
                    break
                except Exception:
                    print(
                        f"\n=== Token limit of {model} reached! ===\n = Trying the next model: {groq_models[index + 1]} =\n")
                    pass

            agent_response = result["output"]
            print(f"\nAgent: {agent_response}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=agent_response))

            if len(chat_history) > 10:
                chat_history = chat_history[-10:]


if __name__ == "__main__":
    main()


## TODO: add a youtube transcription agent