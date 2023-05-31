from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import requests
import os
import getpass
import tiktoken
import re

from langchain.vectorstores import DeepLake

## ENV
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['ACTIVELOOP_TOKEN'] = os.getenv("ACTIVELOOP_TOKEN")
MODEL = "gpt-4"

def num_tokens_from_string(string: str):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def openai_get_content(response):
    return response["choices"][0]["message"]["content"]

## Extarct related context
def extract_relavent_context(program_file):
    code = open(program_file, 'r').read()
    prompt = '''
    Please idenitify and extract all function calls within the following C code.
    Only include those that are not come from the standard C library. Please reply
    me with a list: [function1, function2, ...]. Please do not include any other
    respnoses except the list itself.
    '''
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt + "\n" + code},
        ],
        temperature=0.5,
    )
    return openai_get_content(response)

## Init retriever
def init_retriever():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(dataset_path="hub://xiaowenhu96/doc", read_only=True, embedding_function=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 5

    return retriever


def gen_doc_filter(target):
    pattern = r'.*\s*{}(?!_)'.format(target)
    def filter(x):
        lines = x['text'].data()['value']
        r = False
        for line in lines.split('\n'):
            if re.match(pattern, line):
                r = True
        return r
    return filter
    

def show_docs():
    fns = extract_relavent_context("./simple_udt_no_comments.c").replace("\n", "").replace("[", "").replace("]", "").replace(" ", "").split(",")
    model = ChatOpenAI(model_name=MODEL)
    for fn in fns:
        retriever = init_retriever()
        filter = gen_doc_filter(fn)
        retriever.search_kwargs['filter'] = filter
        qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
        docs = qa._get_docs("Give me gudiance on how to check {}".format(fn), {})
        for doc in docs:
            print(doc.page_content)
            print("Number of tokens:", num_tokens_from_string(doc.page_content))
            print("------------------------------")


def collect_fn_info(program_file):
    fns = extract_relavent_context(program_file).replace("\n", "").replace("[", "").replace("]", "").replace(" ", "").split(",")
    model = ChatOpenAI(model_name=MODEL)
    prompt = '''
    I would like to familiarize myself with the usage standards of certain function calls. For each function, please provide guidance on how to ensure the success of the function call by looking at the RFC and syscalls code, including what return value should be checked. Please provide accurate information and if you are unable to find relevant details in the given context, kindly indicate so in your response. Please answer yes if you understand.
    '''
    chat_history = [((prompt, "Yes! I can help you with that."))]
    results = []
    for fn in fns:
        retriever = init_retriever()
        filter = gen_doc_filter(fn)
        retriever.search_kwargs['filter'] = filter
        qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
        question = "Give me gudiance on how validate the usage of {}".format(fn) 
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        # print(f"-> **Question**: {question} \n")
        # print(f"**Answer**: {result['answer']} \n")
        results.append((fn, result['answer']))
    return results

def analysis_code(program_file):
    code = open(program_file, 'r').read()
    print("Collecting info..")
    fn_infos = collect_fn_info(program_file)
    print("Info collects succeed!")
    prompt = '''
I would like you to simulate a static code analysis system. Your task is to scrutinize various code snippets for potential security vulnerabilities. These might include, but are not limited to, memory leaks, buffer overflows, integer overflows, and unchecked return values. For each code snippet provided, please identify and thoroughly explain any detected security issues, elaborating why they pose a risk. Additionally, recommend appropriate modifications or best practices to mitigate these identified vulnerabilities. 
'''
    prompt_response = '''
Sure! I can help you with that. Please provide me with the code snippets.
    '''
    messages = []
    for fn, info in fn_infos:
        messages.append({"role": "user", "content": "What is the usage of {}?".format(fn)})
        messages.append({"role": "assistant", "content": info})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": prompt_response})
    messages.append({"role": "user", "content": "here is the code:\n" + code})
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    print(response)

analysis_code("./simple_udt_unchecked_return.c")
