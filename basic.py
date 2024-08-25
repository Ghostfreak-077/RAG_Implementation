import os
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import numpy as np
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv


load_dotenv()

## Indexing ##

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

# text embedding
embd = OllamaEmbeddings(model="bge-large")
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)


# Look into cosine similarity. Ideal value is 1
# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)

# similarity = cosine_similarity(query_result, document_result)
# print("Cosine Similarity:", similarity)



#### INDEXING ####

# Load blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

blog_docs = loader.load()


# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=embd)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
docs = retriever.get_relevant_documents("Task Decomposition")
# print(docs)


# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = OllamaLLM(model="llama3:8b")

# chain = prompt | llm
# chain.invoke({"context":docs,"question":"What is Task Decomposition?"})

# prompt_hub_rag = hub.pull("rlm/rag-prompt")
# print(prompt_hub_rag)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What is Task Decomposition?"))