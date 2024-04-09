import json
import os
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores.faiss import FAISS
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = (
            text.replace("```", "")
            .replace("json", "")
            .replace(", ]", "]")
            .replace(", }", "}")
        )
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

api_key = st.session_state.get("api_key", "")

with st.sidebar:
    api_key = st.text_input("OpenAI_API_key", type="password")
    st.session_state["api_key"] = api_key
    if api_key:
        st.caption("API key is set.")
    else:
        st.caption("Please enter your API key ⬆️.")

if api_key == "":
    st.error("Please enter your OpenAI API key")
    st.stop()
else:
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-0125",
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
        api_key=api_key,
    )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions minimum to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

quiz_difficulty = st.selectbox(
                    ":red[Level of Difficulty]",
                    ["1", "2", "3"],
                )

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    os.makedirs(f"./.cache/quiz_files/", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    os.makedirs(f"./.cache/embeddings", exist_ok=True)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="en")
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                    
    Get started by uploading a file or searching on Wikipedia in the sidebar.

    You can choose the difficulty level of the quiz.
    Please select from 1 to 3 in the Level of Difficulty box above.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    generated_quiz_count = len(response["questions"])
    with st.form("questions_form"):
        solved_count = 0
        correct_count = 0
        answer_feedback_box = []
        answer_feedback_content = []

        for index, question in enumerate(response["questions"]):
            st.write(f"{index+1}. {question['question']}")
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
                label_visibility="collapsed",
                key=f"[{quiz_difficulty}]question_{index}",
            )

            answer_feedback = st.empty()
            answer_feedback_box.append(answer_feedback)

            if value:
                solved_count += 1

                if {"answer": value, "correct": True} in question["answers"]:
                    answer_feedback_content.append(
                        {
                            "index": index,
                            "correct": True,
                            "feedback": "Perfect! Correct!!",
                        }
                    )
                    correct_count += 1
                else:
                    answer_feedback_content.append(
                        {
                            "index": index,
                            "correct": False,
                            "feedback": "Oops. Try Again!!",
                        }
                    )

        is_quiz_all_submitted = solved_count == generated_quiz_count

        if is_quiz_all_submitted:
            for answer_feedback in answer_feedback_content:
                index = answer_feedback["index"]
                with answer_feedback_box[index]:
                    if answer_feedback["correct"]:
                        st.success(answer_feedback["feedback"])
                    else:
                        st.error(answer_feedback["feedback"])

        st.divider()

        result = st.empty()

        st.form_submit_button(
            (
                "**:blue[SUBMIT]**"
                if solved_count < generated_quiz_count
                else (
                    "**:yellow[:100: MUY BIEN! CONGRAT!!]**"
                    if correct_count == generated_quiz_count
                    else "**:blue[TRY AGAIN :)]**"
                )
            ),
            use_container_width=True,
            disabled=correct_count == generated_quiz_count,
            args=(True,),
        )

        if correct_count == generated_quiz_count:
                        for _ in range(10):
                            st.balloons()


with st.sidebar:
    st.write(
        "https://github.com/breakerc-J/quiz_gpt/blob/master/pages/03_QuizGPT.py"
    )
   
