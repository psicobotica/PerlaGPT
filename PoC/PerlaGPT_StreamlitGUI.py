"""
PerlaGPT - SteamlitGUI PoC demo version
raul.arrabales | 9 Aug 2023

"""


#####################
# Import Libs
#####################

import os
import sys

from dotenv import load_dotenv, find_dotenv

import requests

import numpy as np
import pandas as pd
import matplotlib

import streamlit as st
from streamlit_chat import message

import openai


#########################
# OpenAI Helper Functions
#########################


##get_completion(prompt, model, temp)
##-----------------------------------
##Helper function to get a GPT 3.5 Turbo completion
##Using the chat completion API
##(https://platform.openai.com/docs/guides/gpt/chat-completions-api)
##
##INPUT:
##- prompt: user's prompt
##- model: OpenAI model (GPT 3.5 Turbo by default)
##- temp: model's temperature (0 by default)
##
##OUTPUT:
##- The instruct LLM response
def get_completion(prompt, model="gpt-3.5-turbo", temp=0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
    )
    return response.choices[0].message["content"]


##*** get_completion_from_messages(messages, model, temp) ***
##        ---------------------------------------------------
##Helper function to get a GPT 3.5 Turbo completion.
##In this case, from the list of past messages.
##Using the chat completion API
##(https://platform.openai.com/docs/guides/gpt/chat-completions-api)
##
##INPUT:
##- messages: list of messages
##- model: OpenAI model (GPT 3.5 Turbo by default)
##- temp: model's temperature (0 by default)
##
##OUTPUT:
##- The instruct LLM response
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temp=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temp,
    )
    return response.choices[0].message["content"]


#####################
# Prompt Engineering
#####################

# Defining the purpose and detailed instructions of the chatbot
system_role_content = f"""
Eres PerlaGPT, una asistente especializada en salud mental cuya misión es \
determinar si una persona tiene un nivel alto de depresión o no. \
Lo primero que tienes que hacer es saludar al usuario, presentarte y explicar \
de forma concisa que le vas a hacer una serie de preguntas para determinar su \
nivel de depresión. Pregunta su nombre para ver cómo dirigirte al usuario. \
Una vez que ya te hayas presentado y le hayas dado la oportunidad de decirte \
su nombre puedes comenzar a hacer las preguntas correspondientes al cuestionario PHQ-9 \
(Patient Health Questionaire ) en su versión en Español. Tienes que hacer \
las preguntas de una en una. No pases a la siguiente pregunta hasta que tengas \
la respuesta de la pregunta actual. No empieces a preguntar en el primer mensaje. \
No muestres todas las preguntas al principio. \
Todas las preguntas se refieren al estado \
emocional de la persona en las últimas dos semanas. Es importante que le \
recuerdes que conteste sólo en relación a las últimas dos semanas. \
Las preguntas del PHQ-9 en español son las siguientes: \
 \
 ¿Con qué frecuencia te has encontrado con poco interés o poco placer en hacer las cosas?  \
 ¿Con qué frecuencia te has sentido decaído/a, deprimido/a o sin esperanzas?  \
 ¿Con qué frecuencia has tenido problemas de sueño (dificultad para quedarte dormido/a o dormir demasiado)?  \
 ¿Con qué frecuencia te has sentido cansado/a o con poca energía?  \
 ¿Con qué frecuencia has estado sin apetito o has comido en exceso?  \
 ¿Con qué frecuencia te has sentido mal contigo mismo/a, que eres un fracaso o que has quedado mal contigo mismo/a o tu familia?  \
 ¿Con qué frecuencia has tenido dificultades para concentrarte en actividades como leer o ver la televisión?  \
 ¿Con qué frecuencia te has movido muy lento o has estado inquieto/a y agitado/a, moviéndote más de lo normal?  \
 ¿Con qué frecuencia has pensado que estarías mejor muerto o en hacerte daño de alguna manera?  \
 \
Si el usuario no entiende bien alguna pregunta, le puedes aclarar su significado \
explicándoselo con otras palabras, intenta utilizar el mismo tipo de lenguaje \
que use el propio usuario. Asegúrate de recoger las respuestas a las nueve preguntas. \
Cada vez que el usuario te responda a una pregunta le puedes mostrar tu \
aprecio por el esfuerzo realizado antes de pasar a la siguiente pregunta. \
Habla siempre con un estilo cercano, comprensivo y amigable, que haga que \
la persona se sienta escuchada, comprendida y respetada. Para cada respuesta \
obtenida debes determinar la frecuencia con la que aparece el síntoma y \
asignarle uno de los siguientes cuatro niveles (los niveles tienen un nombre, \
una descripción y una puntuación numérica asociados): \
 \
- NUNCA: La persona nunca ha experimentado el síntoma \
- VARIOS DIAS: La persona ha experimentado el síntoma algunos días, pero menos de 7 días \
- MÁS DE LA MITAD DE LOS DÍAS: La persona ha experimentado el síntoma más de 7 días \
- CASI TODOS LOS DÍAS: La persona ha experimentado el síntoma siempre o casi siempre \
 \
Si no eres capaz de determinar el nivel correspondiente para alguna de las \
preguntas, debes repetir la pregunta hasta que consigas averiguar la frecuencia \
del síntoma correspondiente. Cuando ya tengas la respuesta a todas las preguntas \
haz un resumen de la información obtenida, indicando de manera muy concisa \
la puntuación correspondiente a cada una de las nueve preguntas.
"""

#####################
# Conversation Context
#####################

# List of messages, starting with system's role
context = [ {'role':'system', 'content':system_role_content} ]



#####################
# Streamlit Settings
#####################

# Hide traceback
st.set_option('client.showErrorDetails', False)

# Setting page title and header
st.set_page_config(page_title="PerlaGPT | PoC Demo", page_icon=":relaxed:")
st.markdown("<h1 style='text-align: center;'>PerlaGPT</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = context


#####################
# OpenAI API Key
#####################

# My OpenAI API key is defined in the local .env file, 
# in the home project dir

_ = load_dotenv('.env')
openai.api_key  = os.environ['OPENAI_API_KEY']

# Check if the API key is valid 
try:
      models = openai.Model.list()
except Exception as e:
      st.error("Error testing API key: {}".format(e))


#########################
# Chat UI
#########################

# container for chat history
response_container = st.container()

# container for text box
input_container = st.container()

with input_container:
    # Create a form for user input
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Tú:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')

    if submit_button and user_input:
        # If user submits input, generate response and store input and response in session state variables
        try:
            # Add user's prompt to the context
            st.session_state['messages'].append({'role':'user', 'content':f"{user_input}"})
            # Add user's prompt to past user's messages
            st.session_state['past'].append(user_input)
            # Get ChatGPT's response based on whole context
            response = get_completion_from_messages(st.session_state['messages'])
            # Add response to context
            st.session_state['messages'].append({'role':'assistant', 'content':f"{response}"})
            # Also add response to generated messages
            st.session_state['generated'].append(response)
        except Exception as e:
            st.error("An error occurred: {}".format(e))

if st.session_state['generated']:
    # Display chat history in a container
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))




