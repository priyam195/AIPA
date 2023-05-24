# AI-Powered Team Productivity Assistant (AIPA)
Project Description: The AI-Powered Team Productivity Assistant (AIPA) is a hackathon project that aims to develop a conversational chatbot using a generative AI large language model (LLM). 
The goal of this project is to create a subject matter expert chatbot that can significantly enhance the productivity of individuals within teams by providing quick and accurate answers to technical questions and facilitating easy access to relevant documents and resources. 
AIPA will learn from the vast amount of documents stored in various team workspaces, including SharePoint, Microsoft Teams, common folders, and other shared locations like Outlook.



## General idea of this project:
* Uploading docs, pdf and wikis as training data in a Database Interface.
* Divide your long text into small chunks that are consumable by GPT.
* Store each chunk in the vector database. Each chunk is indexed by a chunk embedding vector.
* When asking a question to GPT, convert the question to a question embedding vector first.
* Use question embedding vector to search chunks from vector database.
* Combine chunks and questions as a prompt, feed it to GPT.
* Get the response from GPT with the source documents links.
