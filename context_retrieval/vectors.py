from llm import embeddings
from skills_graph import graph
from langchain_neo4j import Neo4jVector





job_retrieval_query="""
    RETURN
    node.title AS text,
    score,
    {
        description: node.description,
        id: node.id
    } AS metadata
    """

occupation_retrieval_query="""
    RETURN
    node.label AS text,
    score,
    {
        description: node.description,
        id: node.id,
        type: node.conceptType
    } AS metadata
    """

skill_retrieval_query="""
    RETURN
    node.label AS text,
    score,
    {
        description: node.description,
        type: node.skillType,
        id: node.id
    } AS metadata
    """

job_title_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="jobTitleEmbeddings",
    embedding_node_property="titleEmbedding",
    text_node_property="title",
    retrieval_query=job_retrieval_query
)

job_description_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="jobDescriptionEmbeddings",
    embedding_node_property="descriptionEmbedding",
    text_node_property="description",
    retrieval_query=job_retrieval_query
)

occupation_label_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="occupationLabelEmbeddings",
    embedding_node_property="labelEmbedding",
    text_node_property="label",
    retrieval_query=occupation_retrieval_query
)

occupation_description_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="occupationDescriptionsEmbeddings",
    embedding_node_property="descriptionEmbedding",
    text_node_property="description",
    retrieval_query=occupation_retrieval_query
)

skill_label_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="skillLabelEmbeddings",
    embedding_node_property="labelEmbedding",
    text_node_property="label",
    retrieval_query=skill_retrieval_query
)

skill_description_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="skillDescriptionEmbeddings",
    embedding_node_property="descriptionEmbedding",
    text_node_property="description",
    retrieval_query=skill_retrieval_query
)



