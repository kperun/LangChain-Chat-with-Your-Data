import os
import openai
import sys

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

some_text = """When writing documents, writers will use document structure to group content. \
    This can convey to the reader, which idea's are related. For example, closely related ideas \
    are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
    Paragraphs are often delimited with a carriage return or two carriage returns. \
    Carriage returns are the "backslash n" you see embedded in this string. \
    Sentences have a period at the end, but also, have a space.\
    and words are separated by space."""

chunk_size = 26
chunk_overlap = 4


def character_splitter():
    '''
    Splits bases on characters, char by char.
    :return:
    '''

    from langchain.text_splitter import CharacterTextSplitter
    c_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    c_splitter.split_text(some_text)


def recursive_splitter():
    '''
    This text splitter is the recommended one for generic text. It is parameterized by a
    list of characters. It tries to split on them in order until the chunks are small enough.
    The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs
    (and then sentences, and then words) together as long as possible, as those would generically seem
    to be the strongest semantically related pieces of text.
    :return:
    '''
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    r_splitter.split_text(some_text)


def token_splitter():
    '''
    Splits a raw text string by first converting the text into BPE tokens, then split these tokens into chunks and
    convert the tokens within a single chunk back into text.
    You can define which tokenizer shall be used
    :return:
    '''
    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
    text_splitter.split_text(some_text)


def markdown_splitter():
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    markdown_document = """# Title\n\n \
    ## Chapter 1\n\n \
    Hi this is Jim\n\n Hi this is Joe\n\n \
    ### Section \n\n \
    Hi this is Lance \n\n 
    ## Chapter 2\n\n \
    Hi this is Molly"""
    # Define for the metadata retrieval how you want it to be named, e.g., extract # as a Header 1.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = md_splitter.split_text(markdown_document)
