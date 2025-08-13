
import os
import sys
import json
from llama_index.legacy.finetuning import (
    generate_qa_embedding_pairs
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llm.adapter import LLMAdapter

load_dotenv()

project_dir = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILES = [os.path.join(project_dir, "../3.txt")]
VAL_FILES = [os.path.join(project_dir, "../3.txt")]

TRAIN_CORPUS_FPATH = os.path.join(project_dir, "data/ft_train_corpus.json")
VAL_CORPUS_FPATH = os.path.join(project_dir, "data/ft_val_corpus.json")

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

llm = DashScope(model_name=DashScopeGenerationModels.QWEN_PLUS,api_key=os.getenv("DASHSCOPE_API_KEY"))

qa_generate_prompt_tmpl = """\
上下文信息如下。

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination in Chinese. The questions should be diverse in nature \
across the document in Chinese. The questions should not contain options, not start with Q1/ Q2. \
Restrict the questions to the context information provided.
Please only output the question itself. Do not include any explanations, prefaces (based on the provided text, create a question like this:), numbers (such as "Question 1:"), quotations, or anything else that is not the question itself.
"""

train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)

train_dataset.save_json(TRAIN_CORPUS_FPATH)
val_dataset.save_json(VAL_CORPUS_FPATH)