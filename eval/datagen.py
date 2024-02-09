import os
import pandas as pd
from datasets import Dataset

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context


def question_gen(pages, bare_template):
    question_schema = ResponseSchema(
        name="question",
        description="a question about the context."
    )

    question_response_schemas = [
        question_schema,
    ]

    question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
    format_instructions = question_output_parser.get_format_instructions()

    question_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")


    qa_template = """\
    You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.

    question: a question about the context.

    Format the output as JSON with the following keys:
    question

    context: {context}
    """

    prompt_template = ChatPromptTemplate.from_template(template=qa_template)

    messages = prompt_template.format_messages(
        context=pages[0],
        format_instructions=format_instructions
    )

    question_generation_chain = bare_template | question_generation_llm

    response = question_generation_chain.invoke({"content" : messages})
    output_dict = question_output_parser.parse(response.content)

    for k, v in output_dict.items():
        print(k)
        print(v)


    qac_triples = []

    for text in pages[9:14]:
        messages = prompt_template.format_messages(
            context=text,
            format_instructions=format_instructions
        )
        response = question_generation_chain.invoke({"content" : messages})
        try:
            output_dict = question_output_parser.parse(response.content)
        except Exception as e:
            continue
        output_dict["context"] = text
        qac_triples.append(output_dict)
    
    return qac_triples


def answer_gen(qac_triples, bare_template):
    answer_generation_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

    answer_schema = ResponseSchema(
        name="answer",
        description="an answer to the question"
    )

    answer_response_schemas = [
        answer_schema,
    ]

    answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
    format_instructions = answer_output_parser.get_format_instructions()

    qa_template = """\
    You are a University Professor creating a test for advanced students. For each question and context, create an answer.

    answer: a answer about the context.

    Format the output as JSON with the following keys:
    answer

    question: {question}
    context: {context}
    """

    prompt_template = ChatPromptTemplate.from_template(template=qa_template)

    messages = prompt_template.format_messages(
        context=qac_triples[0]["context"],
        question=qac_triples[0]["question"],
        format_instructions=format_instructions
    )

    answer_generation_chain = bare_template | answer_generation_llm

    response = answer_generation_chain.invoke({"content" : messages})
    output_dict = answer_output_parser.parse(response.content)

    for k, v in output_dict.items():
        print(k)
        print(v)

    for triple in qac_triples:
        messages = prompt_template.format_messages(
            context=triple["context"],
            question=triple["question"],
            format_instructions=format_instructions
        )
        response = answer_generation_chain.invoke({"content" : messages})
        try:
            output_dict = answer_output_parser.parse(response.content)
        except Exception as e:
            continue
        triple["answer"] = output_dict["answer"]
    
    return qac_triples

def ground_truth_gen(qac_triples):
    ground_truth_qac_set = pd.DataFrame(qac_triples)
    ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
    ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})

    return ground_truth_qac_set


loader = DirectoryLoader("data/")
docs = loader.load()
pages = loader.load_and_split()

for document in docs:
    document.metadata['file_name'] = document.metadata['source']

bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

qac_triples_q = question_gen(pages, bare_template)
qac_triples_qa = answer_gen(qac_triples_q, bare_template)
ground_truth_qac_set = ground_truth_gen(qac_triples_qa)
eval_dataset = Dataset.from_pandas(ground_truth_qac_set)

print(eval_dataset[0])

eval_dataset.to_csv("groundtruth_eval_dataset.csv")