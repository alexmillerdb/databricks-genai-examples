# Databricks notebook source
# MAGIC %md
# MAGIC # Global configuration
# MAGIC
# MAGIC This notebook is meant to be changed once throughout the lifetime of your Agent.
# MAGIC
# MAGIC At a high-level it defines the Unity Catalog locations where the agent, and associated resources, will be stored. Ideally you shouldn't have to change the values in this notebook once you've started building your agent.

# COMMAND ----------

# If you run this notebook directly, you need these dependencies:
# %pip install -U -qqqq databricks-sdk mlflow mlflow-skinny
# dbutils.library.restartPython()

# COMMAND ----------
import json

# COMMAND ----------

user_email = spark.sql("SELECT current_user() as username").collect()[0].username
user_name = user_email.split("@")[0].replace(".", "").lower()[:35]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent configuration
# MAGIC
# MAGIC Important: These notebooks only work on Single User clusters running DBR/MLR 14.3+.
# MAGIC
# MAGIC To begin with, we simply need to configure the following:
# MAGIC 1. `AGENT_NAME`: The name of the Agent.  Used to name the app's Unity Catalog model and is prepended to the output Delta Tables + Vector Indexes
# MAGIC 2. `UC_CATALOG` & `UC_SCHEMA`: [Create a Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/create-catalogs.html#create-a-catalog) and a Schema where the output Delta Tables with the parsed/chunked documents and Vector Search indexes are stored
# MAGIC 3. `UC_MODEL_NAME`: Unity Catalog location to log and store the agent's model
# MAGIC
# MAGIC This notebook will also check that you are using a valid cluster type and all locations / resources exist. Any missing resources will be created.

# COMMAND ----------

# The name of the Agent.  This is used to name the agent's UC model and prepended to the output Delta Tables + Vector Indexes
AGENT_NAME = "agent_app_poc"
RAG_APP_NAME = "Databricks Documentation and E-books Assistant"

# UC Catalog & Schema where outputs tables/indexes are saved
# By default, will use the current user name to create a unique UC catalog/schema & vector search endpoint
# If this catalog/schema does not exist, you need create catalog/schema permissions.
UC_CATALOG = f"main"
UC_SCHEMA = f"alex_m"

## UC Model name where the Agent's model is logged
UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}"

# Vector Search Endpoint where index is loaded
VECTOR_SEARCH_ENDPOINT = f'one-env-shared-endpoint-12'

# You need to create this location and add files
SOURCE_UC_VOLUME = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/source_docs"

# Names of the output Delta Tables tables & Vector Search index

# Delta Table with the parsed documents and extracted metadata
DOCS_DELTA_TABLE = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_docs`"

# Chunked documents that are loaded into the Vector Index
CHUNKED_DOCS_DELTA_TABLE = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_chunked_docs`"

# Vector Index
VECTOR_INDEX_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{AGENT_NAME}_chunked_docs_index"

# Embedding model endpoint. The list of off-the-shelf embeddings can be found here:
# https://docs.databricks.com/en/machine-learning/foundation-models/index.html
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optional configuration
# MAGIC
# MAGIC - `MLFLOW_EXPERIMENT_NAME`: MLflow Experiment to track all experiments for this Agent.  Using the same experiment allows you to track runs across Notebooks and have unified lineage and governance for your Agent.
# MAGIC - `EVALUATION_SET_FQN`: Delta Table where your evaluation set will be stored.  In the POC, we will seed the evaluation set with feedback you collect from your stakeholders.
# MAGIC

# COMMAND ----------

############################
##### We suggest accepting these defaults unless you need to change them. ######
############################

EVALUATION_SET_FQN = f"`{UC_CATALOG}`.`{UC_SCHEMA}`.`{AGENT_NAME}_evaluation_set`"

# MLflow experiment name
# Using the same MLflow experiment for a single app allows you to compare runs across Notebooks
MLFLOW_EXPERIMENT_NAME = f"/Users/{user_email}/{AGENT_NAME}"

# MLflow Run Names
# These Runs will store your initial POC.  They are later used to evaluate the POC model against your experiments to improve quality.

# Data pipeline MLflow run name
POC_DATA_PIPELINE_RUN_NAME = "data_pipeline_poc"
# Chain MLflow run name
POC_CHAIN_RUN_NAME = "agent_poc"

# COMMAND ----------

print("--user info--")
print(f"user_name {user_name}")

print("--agent--")
print(f"AGENT_NAME {AGENT_NAME}")
print(f"UC_CATALOG {UC_CATALOG}")
print(f"UC_SCHEMA {UC_SCHEMA}")
print(f"UC_MODEL_NAME {UC_MODEL_NAME}")

print()
print("--evaluation config--")
print(f"EVALUATION_SET_FQN {EVALUATION_SET_FQN}")
print(f"MLFLOW_EXPERIMENT_NAME {MLFLOW_EXPERIMENT_NAME}")
print(f"POC_DATA_PIPELINE_RUN_NAME {POC_DATA_PIPELINE_RUN_NAME}")
print(f"POC_CHAIN_RUN_NAME {POC_CHAIN_RUN_NAME}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied

w = WorkspaceClient()

# Create UC Catalog if it does not exist, otherwise, raise an exception
try:
    _ = w.catalogs.get(UC_CATALOG)
    print(f"PASS: UC catalog `{UC_CATALOG}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}` does not exist, trying to create...")
    try:
        _ = w.catalogs.create(name=UC_CATALOG)
    except PermissionDenied as e:
        print(
            f"FAIL: `{UC_CATALOG}` does not exist, and no permissions to create.  Please provide an existing UC Catalog."
        )
        raise ValueError(f"Unity Catalog `{UC_CATALOG}` does not exist.")

# Create UC Schema if it does not exist, otherwise, raise an exception
try:
    _ = w.schemas.get(full_name=f"{UC_CATALOG}.{UC_SCHEMA}")
    print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` exists")
except NotFound as e:
    print(f"`{UC_CATALOG}.{UC_SCHEMA}` does not exist, trying to create...")
    try:
        _ = w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"PASS: UC schema `{UC_CATALOG}.{UC_SCHEMA}` created")
    except PermissionDenied as e:
        print(
            f"FAIL: `{UC_CATALOG}.{UC_SCHEMA}` does not exist, and no permissions to create.  Please provide an existing UC Schema."
        )
        raise ValueError(
            "Unity Catalog Schema `{UC_CATALOG}.{UC_SCHEMA}` does not exist."
        )


# COMMAND ----------

# Notebook with the chain's code.  Choose one based on your requirements.  
# If you are not sure, use the `multi_turn_rag_chain`.

# CHAIN_CODE_FILE = "single_turn_rag_chain"

CHAIN_CODE_FILE = "multi_turn_rag_chain"
# Chain configuration
# We suggest using these default settings
rag_chain_config = {
    "databricks_resources": {
        # Only required if using Databricks vector search
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT,
        # Databricks Model Serving endpoint name
        # This is the generator LLM where your LLM queries are sent.
        "llm_endpoint_name": "databricks-meta-llama-3-1-70b-instruct",
    },
    "retriever_config": {
        # Vector Search index that is created by the data pipeline
        "vector_search_index": VECTOR_INDEX_NAME,
        "schema": {
            # The column name in the retriever's response referred to the unique key
            # If using Databricks vector search with delta sync, this should the column of the delta table that acts as the primary key
            "primary_key": "chunk_id",
            # The column name in the retriever's response that contains the returned chunk.
            "chunk_text": "content_chunked",
            # The template of the chunk returned by the retriever - used to format the chunk for presentation to the LLM.
            "document_uri": "doc_uri",
        },
        # Prompt template used to format the retrieved information to present to the LLM to help in answering the user's question
        "chunk_template": "Passage: {chunk_text}\n",
        # The column name in the retriever's response that refers to the original document.
        "parameters": {
            # Number of search results that the retriever returns
            "k": 5,
            # Type of search to run
            # Semantic search: `ann`
            # Hybrid search (keyword + sementic search): `hybrid`
            "query_type": "hybrid",
        },
        # Tag for the data pipeline, allowing you to easily compare the POC results vs. future data pipeline configurations you try.
        "data_pipeline_tag": "poc",
    },
    "llm_config": {
        # Define a template for the LLM prompt.  This is how the RAG chain combines the user's question and the retrieved context.
        "llm_system_prompt_template": """You are a helpful assistant that answers questions based on Databricks documentation and e-books about GenAI, ML, Data Engineering, Security, etc. 
        Use the following pieces of retrieved context to answer the question. 
        Some pieces of context may be irrelevant, in which case you should not use them to form the answer. 
        When responding, cite your source, referring to the passage by the columns in the passage's metadata.

Context: {context}""".strip(),
        # Parameters that control how the LLM responds.
        "llm_parameters": {"temperature": 0.01, "max_tokens": 1500},
    },
    "input_example": {
        "messages": [
            {
                "role": "user",
                "content": "What is RAG?",
            },
        ]
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config & save to YAML
# MAGIC
# MAGIC This step saves the configuration so that the `03_lc_poc_to_review_app` Notebook can use it.

# COMMAND ----------

import yaml
print(f"Using chain config: {json.dumps(rag_chain_config, indent=4)}\n\n Using chain file: {CHAIN_CODE_FILE}")

with open('rag_chain_config.yaml', 'w') as f:
    yaml.dump(rag_chain_config, f)

