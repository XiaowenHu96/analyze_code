import os
import getpass

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


os.environ['ACTIVELOOP_TOKEN'] = os.getenv("ACTIVELOOP_TOKEN")
USERNAME = "" # your username

embeddings = OpenAIEmbeddings(disallowed_special=())

root_dir = './doc'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try: 
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e: 
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

db = DeepLake(dataset_path=f"hub://{USERNAME}/doc", embedding_function=embeddings, public=False) 
db.add_documents(texts)
