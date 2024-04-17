import pymongo
import os, textwrap
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the models 
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
llm=HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature":0.1 ,"max_length":512})
 
@app.route('/',methods=['POST'])

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
    print("==================== query is -",query,"====================")
    # query = 'What is the price of iphone 13?'

    # load pdfs from the Documents directory
    # loader = DirectoryLoader(f'./Documents/', glob="./*.pdf", loader_cls=PyPDFLoader)
    # documents = loader.load()

    # split the documents into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # texts = text_splitter.split_documents(documents)


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
    

   
    # prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # create the chain to answer questions 
    qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})

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
    source = [[item.metadata.get('source')[10:-4], item.metadata.get('page')+1] for item in llm_response['source_documents']]
    print(res)

    index_helpful_answer = res.find("Answer:")
    if index_helpful_answer != -1:  
        helpful_answer_text = res[index_helpful_answer + len("Answer:"):]
        return({"result": helpful_answer_text.strip().replace("\n"," "), "source": source})
    else:
        return("Error")

if __name__ == '__main__':
    main()
