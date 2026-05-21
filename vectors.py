from llm import embeddings, llm
from skills_graph import graph
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

instructions = (
    "Use the given context to answer the question."
    "If you don't know the answer, say you don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

job_title_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="jobTitleEmbeddings",
    embedding_node_property="titleEmbedding",
    text_node_property="title"
)

job_description_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="jobDescriptionEmbeddings",
    embedding_node_property="descriptionEmbedding",
    text_node_property="description"
)

occupation_label_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="occupationLabelEmbeddings",
    embedding_node_property="labelEmbedding",
    text_node_property="label"
)

occupation_description_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="occupationDescriptionsEmbeddings",
    embedding_node_property="descriptionEmbedding",
    text_node_property="description"
)

skill_label_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="skillLabelEmbeddings",
    embedding_node_property="labelEmbedding",
    text_node_property="label",
    retrieval_query="""
    RETURN
    node.label AS text,
    score,
    {
        description: node.description,

        broader_skills: [
            (skill)-[:BROADER_THAN]->(node)
            | [skill.label, skill.description]
        ],

        smaller_skills: [
            (node)-[:BROADER_THAN]->(skill)
            | [skill.label, skill.description]
        ],

        occupations: [
            (occupation)-[:HAS_OPTIONAL_SKILL|HAS_ESSENTIAL_SKILL]->(node)
            | [occupation.label, occupation.description]
        ],

        jobs: [
            (job)-[:REQUIRES]->(node)
            | [job.title, job.description]
        ]
    } AS metadata
    """
)

skill_description_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="skillDescriptionEmbeddings",
    embedding_node_property="descriptionEmbedding",
    text_node_property="description"
)



result = skill_label_vector.similarity_search("breed animals",k=1)
#print(result)

for doc in result:
    print(f"description: {doc.metadata['description']}")
    print(f"Title: {doc.page_content}\n")


question_answer_chain = create_stuff_documents_chain(llm, prompt)

job_title_retriever = create_retrieval_chain(
     job_title_vector.as_retriever(),
     question_answer_chain
)

occupation_label_retriever = create_retrieval_chain(
     occupation_label_vector.as_retriever(),
     question_answer_chain
)

skill_label_retriever = create_retrieval_chain(
     skill_label_vector.as_retriever(),
     question_answer_chain
)

def get_job_title(input):
     return job_title_retriever.invoke({"input": input})

def get_occupation_label(input):
     return occupation_label_retriever.invoke({"input":input})

def get_skill_label(input):
     return skill_label_retriever.invoke({"input":input})