import os
import openai
import sys
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

sys.path.append('../..')
_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

# We are using a local vector db
persist_directory = 'docs/chroma/'

# We are using open AI embeddings, i.e., how queries are converted to vectors
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)


def similarity_search():
    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]
    # We create a new small vector DB to store the three text pieces above
    smalldb = Chroma.from_texts(texts, embedding=embedding)
    question = "Tell me about all-white mushrooms with large fruiting bodies"
    # We request the most similar answers, which might leave out important information as for instance mentioned inthird
    # sentence
    smalldb.similarity_search(question, k=2)
    # With MMR (Max marginal relevance), we can get more diverse retrieved documents
    # Maximum marginal relevance strives to achieve both relevance to the query and diversity among the results.
    smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3)


def metadata_filter():
    # We ask a question which consists of content and metadata (= in which lecture something was mentioned)
    question = "what did they say about regression in the third lecture?"
    docs = vectordb.similarity_search(
        question,
        k=3,
        filter={"source": "docs/cs229_lectures/MachineLearning-Lecture03.pdf"}  # Filter on metadata
    )
    for d in docs:
        print(d.metadata)


def self_query_search():
    """
    We often want to infer the metadata from the query itself.
    To address this, we can use `SelfQueryRetriever`, which uses an LLM to extract:
        1. The `query` string to use for vector search
        2. A metadata filter to pass in as well
    Most vector databases support metadata filters, so this doesn't require any new databases or indexes.
    I.e., we search on both, the content as well as the metadata.
    :return:
    """
    from langchain.llms import OpenAI
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo
    # We define which metadata is available, i.e., based on this metadata we can filter
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`,"
                        " `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]
    # The general information about the content we are searching on
    document_content_description = "Lecture notes"
    llm = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)
    # The retriever combines the llm, the vector db as well as metadata about it
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True
    )
    question = "what did they say about regression in the third lecture?"
    # The retriever creates from the query a filter on the metadata, and within the filtered content
    # it searches via the query
    docs = retriever.get_relevant_documents(question)


def contextual_compression():
    """
    Another approach for improving the quality of retrieved docs is compression.
    Information most relevant to a query may be buried in a document with a lot of irrelevant text.
    Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
    Contextual compression is meant to fix this by compressing the relevant documents to the most relevant parts of it.
    :return:
    """
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    # we need a LLM for compression
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    compressor = LLMChainExtractor.from_llm(llm)
    # the retriever uses compression to make retrieved content smaller for the final LLM answer
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever()
    )
    question = "what did they say about matlab?"
    compressed_docs = compression_retriever.get_relevant_documents(question)


def combining_retrievers():
    """
    Ideally, we combine several techniques to retrieve only relevant but diverse content
    :return:
    """
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    # we need a LLM for compression
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        # We use compression + MMR to get diverse responses
        base_retriever=vectordb.as_retriever(search_type="mmr")
    )
    question = "what did they say about matlab?"
    compressed_docs = compression_retriever.get_relevant_documents(question)
