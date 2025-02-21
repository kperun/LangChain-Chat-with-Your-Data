import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

sys.path.append('../..')
_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
llm_name = "gpt-3.5-turbo"

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
llm = ChatOpenAI(model_name=llm_name, temperature=0)
question = "What are major topics for this class?"

"""
Its possible to track what the different reduction techniques do (to track them) using LangSmith.
If you wish to experiment on the LangSmith platform (previously known as LangChain Plus):
    Go to LangSmith and sign up
    Create an API key from your account's settings
    Use this API key in the code below:
        import os
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
        os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key
    It will track the requests in LangSmith and you can see them. 
"""


def basic_retrieval():
    """
    The retrievers combine retrieval of document + query against LLM. It is using stuffing, i.e., all
    documents are put into one context in the initial query.
    :return:
    """
    # the llm as well as the db against which we go
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever()
    )
    result = qa_chain({"query": question})


def prompting():
    """
    We can define a prompt to combine the documents and the query in a specific way.
    :return:
    """
    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    # Same retriever as before, except for we are using a custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,  # Return the sources for the information you found
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        # chain type not set => stuffing
    )
    result = qa_chain({"query": question})


def map_reduce():
    """
    Instead of stuffing, we use map reduce to combine one answer from several.
    :return:
    """
    qa_chain_mr = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type="map_reduce"  # set to map reduce
    )
    result = qa_chain_mr({"query": question})


def refine():
    """
    Refine sequentially refines the answer step by step from the starting answer.
    :return:
    """
    qa_chain_mr = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type="refine"  # set to refine
    )
    result = qa_chain_mr({"query": question})
