import openai
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from llama_index import LangchainEmbedding
from llama_index import (
    GPTVectorStoreIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    PromptHelper,
    ServiceContext
)
import logging
import sys

openai.api_type = "azure"
openai.api_base = "https://aipaopenairesource.openai.azure.com"
openai.api_version = "2022-12-01"
openai.api_key = "025716b1b48e430e9dc84f85e9fb552a"


llm = AzureOpenAI(deployment_name="AIPA", model_kwargs={
    "api_key": openai.api_key,
    "api_base": openai.api_base,
    "api_type": openai.api_type,
    "api_version": openai.api_version,
})
llm_predictor = LLMPredictor(llm=llm)

embedding_llm = LangchainEmbedding(
    OpenAIEmbeddings(
        model="text-embedding-ada-002",
        deployment="text-embedding-ada-002",
        openai_api_key= openai.api_key,
        openai_api_base=openai.api_base,
        openai_api_type=openai.api_type,
        openai_api_version=openai.api_version,
    ),
    embed_batch_size=1,
)

documents = SimpleDirectoryReader('./knowledge-repository').load_data()


# max LLM token input size
max_input_size = 4000

# set number of output tokens
num_output = 48

# set maximum chunk overlap
max_chunk_overlap = 200

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    embed_model=embedding_llm,
    prompt_helper=prompt_helper
)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()


def answer_user_query(query):
    answer = query_engine.query(query)
    return answer
