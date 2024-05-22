import streamlit as st

import os

# langchain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

from typing import List
from langchain.chains.openai_functions import create_structured_output_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

def EstablecerEntorno(key, uri, user, password):
    os.environ["OPENAI_API_KEY"] = key

    os.environ["NEO4J_URI"] = uri
    os.environ["NEO4J_USERNAME"] = user
    os.environ["NEO4J_PASSWORD"] = password
    print("Entorno establecido")
    LoadGraph()
    print("Grafo construido")
    
class Entities(BaseModel):
  """Identifying information about entities."""

  names: List[str] = Field(
      ...,
      description="All the person or movies appearing in the text",
  )
  
  
def LanzarConsulta(question):
  
  graph = Neo4jGraph()
  graph.refresh_schema()
  
  prompt = ChatPromptTemplate.from_messages(
      [
        (
            "system",
            "Your are extracting person and movies from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
  )

  llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)

  entity_chain = create_structured_output_chain(Entities, llm, prompt)

  match_query = """MATCH (p:Person|Movie)
WHERE p.name CONTAINS $value OR p.title CONTAINS $value
RETURN coalesce(p.name, p.title) AS result, labels(p)[0] AS type
LIMIT 1
"""

  def map_to_database(values):
    result = ""
    for entity in values.names:
      response = graph.query(match_query, {"value": entity})
      try:
        result += f"{entity} maps to {response[0]['result']} {response[0]['type']} in database\n"
      except IndexError:
        pass
      return result

  # Generate Cypher statement based on natural language input
  cypher_template = """Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question. Take into account that the relations are unidirectional and must be respected:
  {schema}
  Entities in the question map to the following database values:
  {entities_list}
  Question: {question}
  Ensure that your Cypher query respects the directionality of the relations in the graph.
  Cypher query:"""

  cypher_prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              "Given an input question, convert it to a Cypher query. No pre-amble.",
          ),
          (
              "human",
              cypher_template
          ),
      ]
  )

  cypher_response = (
      RunnablePassthrough.assign(names = entity_chain)
          | RunnablePassthrough.assign(
              entities_list = lambda x: map_to_database(x["names"]["function"]),
              schema = lambda _: graph.get_schema,
          )
          | cypher_prompt
          | llm.bind(stop=["\nCypherResult:"])
          | StrOutputParser()
      )

  # Cypher validation tool for relationship directions
  corrector_schema = [
      Schema(el["start"], el["type"], el["end"])
      for el in graph.structured_schema.get("relationships")
  ]
  cypher_validation = CypherQueryCorrector(corrector_schema)

  # Generate natural language response based on database results (Que responda en castellano)
  response_template = """Based on the question, Cypher query, and Cypher response, write a natural language response making sure that the answer is in the question language:
  Question: {question}
  Cypher query: {query}
  Cypher Response: {response}"""

  response_prompt = ChatPromptTemplate.from_messages(
      [
          (
              "system",
              "Given an input question an Cypher response, convert it to a natural language answer. No pre-amble.",
          ),
          (
              "human",
              response_template
          ),
      ]
  )
  
  chain = (
      RunnablePassthrough.assign(query=cypher_response)
      | RunnablePassthrough.assign(
          response = lambda x: graph.query(cypher_validation(x["query"])),
      )
      | response_prompt
      | llm
      | StrOutputParser()
  )

  return chain.invoke({"question": question})

# ------------------------------------------------------------------ #
# Load graph
def LoadGraph():
    graph = Neo4jGraph()
    
    movies_query = """
    LOAD CSV WITH HEADERS FROM 'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
    AS row
    MERGE (m:Movie {id:row.movieId})
    SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
    FOREACH (director in split(row.director, '|') |
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
    FOREACH (actor in split(row.actors, '|') |
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
    FOREACH (genre in split(row.genres, '|') |
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
    """
    
    graph.query(movies_query)

# ------------------------------------------------------------------ #
# Streamlit page
st.title('Consultor de grafos en Neo4j')

multi = '''Este consultor trata con un grafo sobre películas donde podrás encontrar directores, actores, géneros y, por supuesto, películas. Estos son los atributos que contiene cada nodo:
- Persona: Nombre
- Película: Fecha de estreno, título y puntuación IMDB
- Género: nombre
'''

st.image('EjemploGrafo')
        
st.markdown(multi)

with st.sidebar.form('entorno'): 
    # Environment configuration
    key = st.text_input('OpenAI API key', type='password')

    st.subheader('Configuración del entorno Neo4j')
    uri = st.text_input('Uri: ', 'bolt://')
    user = st.text_input('Usuario: ', 'neo4j')
    password = st.text_input('Contraseña: ', type='password')

    establecerEntorno = st.form_submit_button('Establecer')
    
    if (establecerEntorno):
        EstablecerEntorno(key, uri, user, password)
        
    if not key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key', icon='⚠')

# Entrada de texto que ocupa el ancho de la página
# st.subheader('Consultas')
#question = st.text_area('Ingrese la consulta a realizar al grafo', '')


with st.form('my_form'):
    text = st.text_area('Consulta:', '')
    submitted = st.form_submit_button('Consultar')
    
    if submitted and key.startswith('sk-'):
        #st.info("Respuesta:")
        with st.spinner('Wait for it...'):
            st.info(LanzarConsulta(text))
