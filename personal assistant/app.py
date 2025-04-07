import streamlit as st
from prompts import game_dev, simple_coder, data_scientist, planner
from groq import Groq
import regex as re
import ollama
import time
import os


agent = Groq(api_key=os.environ.get("GROQ_API_KEY"))
groq_agent = agent

groq_llms = ["llama-3.3-70b-versatile", "qwen-2.5-32b", "qwen-2.5-coder-32b"]
local_llms =["qwen2.5-coder:7b-instruct","llama3.2:latest", "mistral:latest", "deepseek-r1:7b", "hf.co/Triangle104/Dolphin3.0-Llama3.1-8B-Q5_K_S-GGUF:latest", "gemma2:9b"]

simple = "you are a useful universal assistant. Answer a question only on point but as accurate as possible"
modified_prompt = ""
agent_class = [game_dev,
               simple_coder,
               simple,
               data_scientist,
               planner,
               modified_prompt]
available_classes = ["game_dev", "simple_coder", "simple", "data_scientist", "planner", "modified_prompt"]

all_commands = ["", "-q", "-py", "-cc", "-p", "-r", "-show_file", "-file", "-set", "-play", "-over", "-show_llms", "-menu"]
short_explanations = {"-cc": "refresh conversation",
                      "-file": "print main file",
                      "-set *file name*": "set file as main",
                      "-show_file": "print file content",
                      "-r *query*": "load file to LLM context window",
                      "-py": "create .py file",
                      "-over": "overwrite the main file",
                      "-play": "run the main file",
                      "-show_llms": "show all available LLMs and specialisations",
                      "-change_llm 0 0": "change LLM",
                      "-change_prompt 0": "change specialisation"}
short_list = {"-cc": "refresh conversation",
                      "-file": "print main file",
                      "-set *file name*": "set file as main",
                      "-show_file": "print file content",
                      "-r *query*": "load file to LLM context window",
                      "-py": "create .py file",
                      "-over": "overwrite the main file",
                      "-play": "run the main file"}

###################         Changable           ######################

Agent = [groq_llms[0], agent_class[2]]  #   the specialisation also down set to 2!

###################         Changable           ######################


                        ###         Application
                        #   style
st.set_page_config(page_title="Personal Assistant",layout="wide",initial_sidebar_state="auto")


                        ###         Setting the session states          ###
#   saving the Agent in session_state
if "Agent" not in st.session_state:
    st.session_state.Agent = Agent

#   setting specialisation of the agent
if "selection" not in st.session_state:
    st.session_state.selection = 2

#   initializing file "location"
if "filename" not in st.session_state:
    st.session_state.filename = ""

#   initial message memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": Agent[1]})

#   initial system message memory
if "system_messages" not in st.session_state:
    st.session_state.system_messages = []

if "activate_clicked" not in st.session_state:
    st.session_state.activate_clicked = False

#   pages
with st.sidebar:
    st.sidebar.title("-= Welcome Home =-")
    st.header(f"***talking to***: \n\n**{st.session_state.Agent[0]}**")
    st.subheader(f"***specialisation***: \n\n**{available_classes[st.session_state.selection]}**")

    options = ["Info", "Prompt center"]
    selected_page = st.sidebar.selectbox("", options)
    if selected_page == "Info":
        functions_list = "\n".join(
            [f"- **{function}**: {explanation}" for function, explanation in short_list.items()])
        system_info = (f"===========  Main Functions  ==========\n\n"
                       f"{functions_list}\n\n"
                       f"===================================\n\n"
                       f"the local LLMs can be downloaded on\n"
                       f"ollama.com. To use Groq-LLMs, API as\n"
                       f"system variable required (https://groq.com/groqcloud/)")
        st.markdown(system_info)
    if selected_page == "Prompt center":

        prompt_container = st.container(height=300)
        master_prompt = st.text_area("enter your prompt", value="<role></role>"
                      "<instruction></instruction>"
                      "<example></example>")
        with prompt_container:
            if master_prompt:
                st.write(master_prompt)
        erase_memory = st.checkbox("erase memory")
        if erase_memory:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": st.session_state.Agent[1]})
        if st.button("activate"):
            st.session_state.activate_clicked = True
            st.write("prompt activated!")
            modified_prompt = master_prompt
            st.session_state.Agent[1] = modified_prompt
            st.session_state.selection = 5
            st.session_state.messages.append({"role": "system", "content": st.session_state.Agent[1]})
        system_info = (f"\nget creative with prompts! for example:\n\n"
                       f"-  language tutor:\n\n"
                       f"<role>spanish language tutor</role> \n"
                       f"<instruction>speak only in spanish</instruction> \n"
                       f"<instruction>speak a relatively simple language</instruction> \n"
                       f"<instruction>if i do a mistake, correct me</instruction>\n\n"
                       f"-  moody companion:\n\n"
                       f"<role>you are my moody companion</role> \n"
                       f"<instruction>try to be cheerful and lift my mood</instruction> \n"
                       f"<instruction>you can tell jokes time to time</instruction>\n\n"
                       f"-  writer\n\n"
                       f"<role>you are a intelligent writer</role>\n"
                       f"<instruction>speak in a clear structure: introduction, main content, ending</instruction>\n"
                       f"<instruction>use a high quality vocabulary</instruction>\n"
                       f"-  doctor\n\n"
                       f"<role>you are a doctor</role>\n "
                       f"<instruction>your goal is to take FULL anamnesis</instruction>\n"
                       f"<instruction>Please think through, do the questionary step by step</instruction>\n"
                       f"<instruction>when you know what the disease is, list all the exact symptoms and the diagnosis</instruction>\n"
                       f"<instruction>use ONLY short answers / questions</instruction>\n")
        st.markdown(system_info)


col1, col2 = st.columns(2)
with col1:
    st.subheader("Personal Assistant")
    message_container = st.container(height=600, border=True)

with col2:
    st.subheader("System Response")
    system_container = st.container(height=600, border=True)


                    ###             all functions           ###
def copy_code(message):
    pattern = r'```python(.*?)```'
    match = re.search(pattern, message, re.DOTALL)
    if match:
        code = match.group(1)
        return code.strip()
    else:
        return None
def create_file(filename):
    def get_code():
        return copy_code(st.session_state.messages[-1]["content"])

    def create(filename):
        code = get_code()
        if code is None:
            with system_container:
                system_info = f"\nOoops! The code not found!."
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
                return None
        try:
            with open(filename, 'w') as f:
                f.write(code)
            return filename
        except Exception:
            with system_container:
                system_info = f"\nOoops! Something went wrong!."
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
                return None

    base = f"./test/{filename}.py"
    counter = 0
    while True:
        filename = f"{os.path.splitext(base)[0]}{f'({counter})' if counter else ''}{os.path.splitext(base)[1]}"
        if not os.path.exists(filename):
            result = create(filename)
            if result is not None:
                with system_container:
                    system_info = f"\npython file {os.path.basename(filename)[:-3]}.py created!"
                    st.markdown(system_info)
                    st.session_state.system_messages.append({"role": "system", "content": system_info})
                return os.path.basename(filename)[:-3]
            return None
        counter += 1
def read_python_file(filename):
    try:
        if os.path.exists(f"./test/{filename}.py"):
            with open(f"./test/{filename}.py", 'r') as file:
                content = file.read()
            return content
    except FileNotFoundError:
        return None
def show_file(filename):
    try:
        if os.path.exists(f"./test/{filename}.py"):
            with open(f"./test/{filename}.py", 'r') as file:
                content = file.read()
            return content
    except FileNotFoundError:
        return None
def set_file(filename):
    try:
        with open(f"./test/{filename}.py", 'r') as file:
            file.read()
            with system_container:
                system_info = f"Main file set to: {filename}.py"
                st.session_state.filename = filename
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
        return st.session_state.filename
    except FileNotFoundError:
        with system_container:
            system_info = f"File {filename}.py not found."
            st.markdown(system_info)
            st.session_state.system_messages.append({"role": "system", "content": system_info})
        with open(f"./test/{filename}.py", 'w') as f:
            f.write("-=created by Agent=-")
        with system_container:
            system_info = f"\nNew file with a name {filename}.py created and set to main file."
            st.markdown(system_info)
            st.session_state.filename = filename
            st.session_state.system_messages.append({"role": "system", "content": system_info})
        return st.session_state.filename
def change_LLM(list, Agent):
    try:
        if list[1] == "0":
            Agent = [groq_llms[int(list[2])], Agent[1]]
            with system_container:
                st.write(f"#### User request: {" ".join(list)}")
                st.session_state.system_messages.append({"role": "system", "content": f"#### User request: {" ".join(list)}"})
                system_info = f"the LLM changed to {Agent[0]}"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            return Agent
        if list[1] == "1":
            Agent = [local_llms[int(list[2])], Agent[1]]
            with system_container:
                st.write(f"#### User request: {" ".join(list)}")
                st.session_state.system_messages.append({"role": "system", "content": f"#### User request: {" ".join(list)}"})
                system_info = f"the LLM changed to {Agent[0]}"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            return Agent
        else:
            with system_container:
                st.write(f"#### User request: {" ".join(list)}")
                st.session_state.system_messages.append({"role": "system", "content": f"#### User request: {" ".join(list)}"})
                system_info = (f"something went wrong!"
                               f"the agent is left as {Agent[0]}")
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            return Agent

    except Exception as e:
        with system_container:
            st.write(f"#### User request: {" ".join(list)}")
            system_info = (f"something went wrong! try again")
            st.markdown(system_info)
            st.session_state.system_messages.append({"role": "system", "content": system_info})
            return Agent
def change_prompt(list, Agent):
    try:
        if agent_class[int(list[1])] in agent_class:
            Agent = [Agent[0], agent_class[int(list[1])]]
            with system_container:
                system_info = f"the specialisation changed to {available_classes[int(list[1])]}"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            return Agent, list
        else:
            with system_container:
                system_info = "something went wrong! try again"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            return None
    except Exception as e:
        with system_container:
            system_info = "something went wrong! try again"
            st.markdown(system_info)
            st.session_state.system_messages.append({"role": "system", "content": system_info})
        return None
def llm_answer(prompt, llm):
    try:
        st.session_state.messages.append(
            {"role": "user", "content": prompt})
        message_container.chat_message("user").markdown(prompt)
        with message_container.chat_message("assistant"):
            with st.spinner("Assistant thinks..."):
                response = ""
                if llm in local_llms:
                    stream = ollama.chat(model=llm, messages=st.session_state.messages, stream=True)
                    for chunk in stream:
                        content = chunk["message"]["content"]
                        response += content
                        print(content, end="")  # flow like conversation feels better, for streamlit works only in cmd!
                    st.write(response)
                if llm in groq_llms:
                    groq = groq_agent.chat.completions.create(messages=st.session_state.messages, model=llm)
                    response = groq.choices[0].message.content
                    for flow in response:
                        print(flow, end="")
                        time.sleep(0.002)  # only for some cosmetics..
                    st.write(response)
        return response
    except Exception as e:
        st.error(e)
def special_answer(prompt, context, llm):
    try:
        st.session_state.messages.append(
            {"role": "user", "content": prompt})
        message_container.chat_message("user").markdown(prompt)
        with message_container.chat_message("assistant"):
            with st.spinner("Assistant thinks..."):
                response = ""
                if llm in local_llms:
                    stream = ollama.chat(model=llm, messages=[st.session_state.messages[0], {"role": "user", "content": f"answer the question: {prompt}, using the given context: <context>{context}</context>"}], stream=True)
                    for chunk in stream:
                        content = chunk["message"]["content"]
                        response += content
                        print(content, end="")  # flow like conversation feels better, for streamlit works only in cmd!
                    st.write(response)
                if llm in groq_llms:
                    groq = groq_agent.chat.completions.create(messages=[st.session_state.messages[0], {"role": "user", "content": f"answer the question: {prompt}, using the given context: <context>{context}</context>"}], model=llm)
                    response = groq.choices[0].message.content
                    for flow in response:
                        print(flow, end="")
                        time.sleep(0.002)  # only for some cosmetics..
                    st.write(response)
        return response
    except Exception as e:
        st.error(e)


                    ###             main functions          ###
def menu(prompt, Agent):

    #   delete chat history
    if prompt == "-cc":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"#### User request: {prompt}")
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": Agent[1]})
            system_info = f"===== conversation history successfully cleaned ===== "
            st.write(system_info)
            st.session_state.system_messages.append({"role": "system", "content": system_info})
    #   load .py file (if it is set already with a -set function) into context window and make a special query on it

    #   show content of a file that is set to a working place
    if prompt == "-show_file":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"#### User request: {prompt}")
            content = show_file(st.session_state.filename)
        if content is not None:
            with system_container:
                system_info = f"```python{content}```"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
        else:
            with system_container:
                system_info = f"\nThe main file is not set.\n"
                st.write(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
    if "-set" in prompt[:4]:
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"##### User request: {prompt}")
        if len(prompt.split(" ")) == 1:
            pass
        if len(prompt.split(" ")) > 2:
            with system_container:
                system_info = f"oops! something went wrong here"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
        else:
            st.session_state.filename = prompt.split(" ")[-1]
            st.session_state.filename = set_file(st.session_state.filename)
            prompt = ""

    #   load main file to context window and answer exclusively on it. (the file does not save to main conversation memory, only response)
    if "-r" in prompt[:3]:
        prompt = prompt[2:]
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"##### User request: {prompt}")
            content = read_python_file(st.session_state.filename)
            if content == None:
                system_info = f"\nFile '{st.session_state.filename}.py' not found. Please set a main file"
                prompt = ""
                return prompt, system_info
            else:
                system_info = (f"\nthe file {st.session_state.filename}.py was read\n"
                               f"```python{show_file(st.session_state.filename)}```")
        with system_container:
            st.write(system_info)
        with message_container:
            prompt = prompt
            response = special_answer(prompt, content, st.session_state.Agent[0])
            st.session_state.messages.append({"role": "assistant", "content": response})
            prompt = ""
        return prompt, system_info

    #   show which file is set to a working place
    if prompt == "-file":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"##### User request: {prompt}")
            if st.session_state.filename == "":
                system_info = f"The main file not set"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            else:
                system_info = f"The main file set to: {st.session_state.filename}.py"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})

    #   run file that is set
    if prompt == "-play":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"##### User request: {prompt}")
            import subprocess
            def run_script(filename):
                try:
                    result = subprocess.run(["python", f"./test/{filename}.py"], capture_output=True, text=True)
                    return result.stdout if result.returncode == 0 else result.stderr
                except Exception as e:
                    return str(e)

            st.code(run_script(st.session_state.filename), language="python")
            st.session_state.system_messages.append({"role": "system", "content": f"```python{run_script(st.session_state.filename)}```"})

    #   create a .py file from last answer
    if prompt == "-py":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"##### User request: {prompt}")
        if st.session_state.filename == "":
            st.session_state.filename = create_file("code_test")
        else:
            st.session_state.filename = create_file(st.session_state.filename)

    #   overwrite the .py file that is set
    if prompt == "-over":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"##### User request: {prompt}")
        with open(f"./test/{st.session_state.filename}.py", 'w') as f:
            f.write(copy_code(st.session_state.messages[-1]["content"]))
            with system_container:
                system_info = f"\npython file {st.session_state.filename}.py was changed!"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})

    #   show current and possible LLMs and specialisations
    if prompt == "-show_llms":
        with system_container:
            st.write(f"##### User request: {prompt}")
            system_info = (f"- **using LLM**: {st.session_state.Agent[0]}\n"
                           f"- **groq LLMs**: {groq_llms}\n"
                           f"- **ollama LLMs**: {local_llms}\n"
                           f"===============================================\n"
                           f"- to change LLM, type for excample: '-change_llm 0 0' (first number stand for line, 2nd for number of LLM)\n\n"
                           f"===============================================\n"
                           f"- using **specialisation**: {available_classes[agent_class.index(st.session_state.Agent[1])]}\n"
                           f"- **available specialisations**: {available_classes}\n"
                           f"- to change specialisations, type for excample: '-change_prompt 0' (0 stands for first element in a list)\n"
                           f"- NOTE: changes for specialisation will restart the conversation automatically\n")
            st.markdown(system_info)
            st.session_state.system_messages.append({"role": "system", "content": system_info})

    #   change LLM
    if "-change_llm" in prompt:
        if len(prompt) > 15:
            with system_container:
                st.write(f"##### User request: {prompt}")
                st.session_state.system_messages.append({"role": "system", "content": prompt})
                system_info = "something went wrong! try again"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
        else:
            selection = prompt.split(" ")
            st.session_state.Agent = change_LLM(selection, st.session_state.Agent)
            prompt = ""

    #   change specialisation
    if "-change_prompt" in prompt:
        with system_container:
            st.write(f"##### User request: {prompt}")
            st.session_state.system_messages.append({"role": "system", "content": prompt})
            if len(prompt) > 16:
                system_info = "something went wrong! try again"
                st.markdown(system_info)
                st.session_state.system_messages.append({"role": "system", "content": system_info})
            else:
                selection = prompt.split(" ")
                Agent, selection = change_prompt(selection, st.session_state.Agent)
                st.session_state.selection = int(selection[1])
                st.session_state.messages = []
                st.session_state.messages.append({"role": "system", "content": Agent[1]})
                st.markdown(f"conversation restarted")
        prompt = ""

    #   quit the app
    if prompt == "-q":
        quit()

    #   show main functions
    if prompt == "-menu":
        st.session_state.system_messages.append({"role": "user", "content": prompt})
        with system_container:
            st.write(f"#### User request: {prompt}")

        with system_container:
            functions_list = "\n".join(
                [f"- **{function}**: {explanation}" for function, explanation in short_explanations.items()])

            system_info = (f"=====  Available Functions  =====\n\n"
                           f"{functions_list}\n\n"
                           f"=====================================")
            st.markdown(system_info)
            return prompt, system_info

    return prompt, None

def main():
    for message in st.session_state.messages[1:]:
        with message_container.chat_message(message["role"]):
            st.markdown(message["content"])
    for system_message in st.session_state.system_messages:
        with system_container.chat_message(system_message["role"]):
            st.write(f"{system_message["content"]}")

    if prompt := st.chat_input("User prompt: "):
        prompt, system_info = menu(prompt, Agent)
        if system_info is not None:
            st.session_state.system_messages.append({"role": "system", "content": system_info})


        if prompt in all_commands:
            return ""

        response = llm_answer(prompt, st.session_state.Agent[0])
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
