import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader, FileSystemBlobLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader

sys.path.append('../..')
_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']


def load_pdf():
    # Load a PDF from local folder
    loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    pages = loader.load()

    # Print the content of the first page
    page = pages[0]
    print(page.page_content[0:500])
    # Print the metadata
    print(page.metadata)


def load_youtube():
    # Load a youtube video, transcript it using OpenAIWhisper
    url = "https://www.youtube.com/watch?v=jGwO_UgTS7I"
    save_dir = "docs/youtube/"
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),  # fetch from youtube
        # FileSystemBlobLoader(save_dir, glob="*.m4a"),   #fetch locally
        OpenAIWhisperParser()
    )
    docs = loader.load()
    print(docs[0].page_content[0:500])


def load_web():
    loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/titles-for-programmers.md")
    docs = loader.load()
    print(docs[0].page_content[:500])
