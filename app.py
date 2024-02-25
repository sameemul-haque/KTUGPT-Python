import os, textwrap
from pprint import pprint
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

def main():
    # load env
    load_dotenv()

    # load pdfs from the Documents directory
    loader = DirectoryLoader(f'./Documents/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    # create the retriever
    db_instructEmbedd = FAISS.from_documents(texts, instructor_embeddings)
    retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
    query = 'What is operating system?'
    # print('retriever search type:',retriever.search_type)  # retriever search type is similarity search
    # print('retriever search kwargs:',retriever.search_kwargs)
    # docs = retriever.get_relevant_documents(query)
    # pprint(docs[0])
    # pprint(docs[1])
    # pprint(docs[2])

    # Initialize the model  falcon-7b
    os.environ["HUGGINGFACEHUB_API_TOKEN"]
    llm=HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.1 ,"max_length":512})

    # create the chain to answer questions 
    qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    ## Cite sources
    def wrap_text_preserve_newlines(text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')
        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

    def process_llm_response(llm_response):
        print(wrap_text_preserve_newlines(llm_response['result']))
        print('\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])

    llm_response = qa_chain_instrucEmbed(query)
    process_llm_response(llm_response)

if __name__ == '__main__':
    main()
