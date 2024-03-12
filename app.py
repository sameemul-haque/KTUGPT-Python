import pymongo
import os, textwrap
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from flask import Flask, request

app = Flask(__name__)

@app.route('/',methods=['GET'])

def main():
    # load env
    load_dotenv()
    mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # connect to mongodb
    client = pymongo.MongoClient(mongodb_connection_string)
    db = client.test_database
    collection = db.textbooks

    query = request.args.get('q')
    # query = unquote(query)

    # load pdfs from the Documents directory
    # loader = DirectoryLoader(f'./Documents/', glob="./*.pdf", loader_cls=PyPDFLoader)
    # documents = loader.load()

    # split the documents into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_documents(documents)

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # create the retriever
    # db_instructEmbedd = FAISS.from_documents(texts, instructor_embeddings)
    # retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
    # retriever search type is similarity search

    # # create the retriever and do embedding
    # vector_search = MongoDBAtlasVectorSearch.from_documents(
    #     documents=texts,
    #     embedding=instructor_embeddings,
    #     collection=collection,
    #     index_name="default",
    # )

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        mongodb_connection_string,
        "test_database" + "." + "textbooks",
        instructor_embeddings,
        index_name="default",
    )
    retriever = vector_search.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    
    # query = 'What is operating system?'

    # Initialize the model  falcon-7b
    llm=HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1 ,"max_length":512})

    # create the chain to answer questions 
    qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')
        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

    llm_response = qa_chain_instrucEmbed(query)
    res = wrap_text_preserve_newlines(llm_response['result'])
    # print(res)

    index_helpful_answer = res.find("Helpful Answer:")
    if index_helpful_answer != -1:  
        helpful_answer_text = res[index_helpful_answer + len("Helpful Answer:"):]
        return(helpful_answer_text.strip())
    else:
        return("Error")

if __name__ == '__main__':
    main()
