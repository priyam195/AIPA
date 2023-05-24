from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from llama_index import LangchainEmbedding
from pathlib import Path
import  openai
import os
import json
import openai
import logging
import sys
#import fitz
import docx2txt
import aspose.words as aw
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper,
    ServiceContext
)
import glob
import shutil



LOAD_DATA_DIR = "C:\code/newOpen\OpenAI\OpenAI\knowledge-repository"
TMP_DIR = "C:\code/newOpen\OpenAI\OpenAI/tmp"
TXT_WRITE_DIR = "C:\code/newOpen\OpenAI\OpenAI\ConvertToTextFile"
PERSIST_DIRECTORY = "C:\code/newOpen\OpenAI\OpenAI/Store"




openai.api_type = "azure"
openai.api_base = "https://aipaopenairesource.openai.azure.com"
openai.api_version = "2022-12-01"
#os.environ["OPENAI_API_KEY"] = "<insert api key from azure>"
openai.api_key = os.getenv("OPENAI_API_KEY")


llm = AzureOpenAI(deployment_name="AIPA", model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
})


embedding_llm = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="text-embedding-ada-002",
        openai_api_key= openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
        chunk_size = 1
    )



def delete_all_files_in_directory(dir_path):
    files = glob.glob(dir_path)
    for f in files:
        os.remove(f)

def create_dir_if_not_present(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

create_dir_if_not_present(TXT_WRITE_DIR)
create_dir_if_not_present(PERSIST_DIRECTORY)
def getSourceFileName(source):
    json_str= source.split("metadata=")[-1]
    return json_str

def docs_to_txt(path):
    directory = glob.glob(LOAD_DATA_DIR+'/*.docx')
    for file_name in directory:
        with open(file_name, 'rb') as infile:
            outfile = open(TXT_WRITE_DIR+"/"+file_name[:-5]+'.txt', 'w', encoding='utf-8')
            doc = docx2txt.process(infile)

        outfile.write(doc)

    outfile.close()
    infile.close()
    

#-------------------------------------------------------------------------

source_dict = {}
for path in glob.iglob(f'{LOAD_DATA_DIR}/*'):
    filenameWithoutExt = Path(path).stem
    if path.__contains__(".docx"):
        doc = aw.Document(path)     
        saveFilePath = TXT_WRITE_DIR+"/"+filenameWithoutExt+".txt"
        doc.save(saveFilePath)
        source_dict[filenameWithoutExt]=filenameWithoutExt+".docx"
        # with open(path, 'rb') as infile:
        #     outfile = open(filenameWithoutExt+'.txt', 'w', encoding='utf-8')
        #     doc = docx2txt.process(infile)
        #     outfile.write(doc)
        # outfile.close()
        # infile.close()

    if path.__contains__(".pdf"):
        #Todo
        file1 = open(TXT_WRITE_DIR+"/"+"filenameWithoutExt.txt","a")
        # doc = fitz.open('sample.pdf') 
        # text = ""
        # for page in doc:
        #     text+=page.get_text()
        # file1.write(text)
        # file1.close()
        
    if path.__contains__(".txt"):
        doc = aw.Document(path)
        saveFilePath = TXT_WRITE_DIR+"/"+filenameWithoutExt+".txt"
        doc.save(saveFilePath)
        source_dict[filenameWithoutExt]=filenameWithoutExt+".txt"
    

        


        
    


docs_list = []
for path in glob.iglob(f'{TXT_WRITE_DIR}/*.txt'):
    print(path)
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=documents, embedding=embedding_llm, persist_directory=PERSIST_DIRECTORY)
    vectordb.persist()
    vectordb = None



vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_llm)


#os.rmdir(TXT_WRITE_DIR)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

chat_history = []
# query = """Hey AIPA, I am a new hire in the team and there is a lot of information to absorb. Sometimes I feel overwhelmed with so many Wikis, SOPs, Policies, presentations and reports. 
# I want to speed up my onboarding process but on the same time I don’t want to annoy my teammates with stupid questions every now and then. 
# How can you help me here"""
# result = qa({"question": query, "chat_history": chat_history})
# chat_history.append((query, result["answer"]))
# print('query was:', query)
# print('answer was:', result["answer"])
# print(source_dict[Path(result['source_documents'][0].metadata['source']).stem])
# print("--------------------------------------")


# query = "Awesome, can you tell me what the current GPU cycle time is as compared to general purpose compute ?"
# result = qa({"question": query, "chat_history": chat_history})
# print('query was:', query)
# print('answer was:', result["answer"])
# print(source_dict[Path(result['source_documents'][0].metadata['source']).stem])
# chat_history.append((query, result["answer"]))
# print("--------------------------------------")


# query = "I am an operations manager in the cloud supply chain and want to know if I can shift the capacity demand from Mumbai region to Hyderabad ?"
# result = qa({"question": query, "chat_history": chat_history})
# print('query was:', query)
# print('answer was:', result["answer"])
# print(source_dict[Path(result['source_documents'][0].metadata['source']).stem])
# print("--------------------------------------")

# if len(chat_history) >= 3:
#     chat_history.pop(0)
# print(len(chat_history))


#clear 
# delete_all_files_in_directory(PERSIST_DIRECTORY)
# delete_all_files_in_directory(TXT_WRITE_DIR)


#clear
#shutil.rmtree(TMP_DIR)
#shutil.rmtree(TXT_WRITE_DIR)


def answer_user_query(query):
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
     
    return [result["answer"],source_dict[Path(result['source_documents'][0].metadata['source']).stem]]