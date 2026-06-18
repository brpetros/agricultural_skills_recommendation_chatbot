from llm import embeddings
from graph_db import graph
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

# retrieval query for all Searchable nodes
retrieval_query ="""
RETURN
CASE
    WHEN 'Job' IN labels(node) THEN coalesce(node.title, node.id)
    WHEN 'Region' IN labels(node) THEN coalesce(node.location,"Unknown")
    WHEN 'Skill' IN labels(node) THEN coalesce(node.label, node.id)
    WHEN 'Occupation' IN labels(node) THEN coalesce(node.label, node.id)
    ELSE coalesce(node.id, "Unknown")
END AS text,

score,

{
    id: coalesce(node.id, ""),
    type: CASE
        WHEN 'Job' IN labels(node) THEN 'Job'
        WHEN 'Region' IN labels(node) THEN 'Region'
        WHEN 'Skill' IN labels(node) THEN 'Skill'
        WHEN 'Occupation' IN labels(node) THEN 'Occupation'
        ELSE 'Unknown'
    END,
    description: coalesce(node.description, "")
} AS metadata
"""

# the following vectors will be removed - replaced by a single vector accross all nodes (search_vector)
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

job_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="jobEmbeddings",
    embedding_node_property="unifiedEmbedding",
    text_node_property="title",
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

occupation_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="occupationEmbeddings",
    embedding_node_property="unifiedEmbedding",
    text_node_property="label",
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

skill_vector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="skillEmbeddings",
    embedding_node_property="unifiedEmbedding",
    text_node_property="label",
    retrieval_query=skill_retrieval_query
)"""

search_vector = Neo4jVector.from_existing_index(
    embedding=embeddings,
    graph=graph,
    index_name="searchEmbeddings",
    embedding_node_property="unifiedEmbedding",
    text_node_property="searchText",
    retrieval_query=retrieval_query
)