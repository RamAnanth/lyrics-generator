import cohere
import streamlit as st

CO_API_KEY = st.secrets["CO_API_KEY"] # Retrieve Cohere API Key 

# if "CO_API_KEY" not in os.environ:
#     raise KeyError("CO_API_KEY not found in st.secrets or os.environ. Please set it in "
#                    ".streamlit/secrets.toml or as an environment variable.")

def generate(co, prompt):
    response = co.generate(  
    model='xlarge',  
    prompt = prompt,  
    max_tokens=100,  
    temperature=0.98,  
    stop_sequences=["--"])

    gens = response.generations
    return gens

def command(co, prompt, model='command-xlarge-20221108', *args, **kwargs):
    gens = co.generate(
    prompt, 
    model=model, 
    max_tokens=100, 
    return_likelihoods='GENERATION', 
    temperature=0.98
    *args, **kwargs)
    return sorted(gens, key=lambda g: -g.likelihood)    

initial_prompt = f"""  
This program generates lyrics for a song given the genre.

Genre:Rap 
Lyrics:You better lose yourself in the music
The moment, you own it, you better never let it go (Go)
You only get one shot, do not miss your chance to blow
This opportunity comes once in a lifetime, yo
You better lose yourself in the music
The moment, you own it, you better never let it go (Go)
You only get one shot, do not miss your chance to blow
This opportunity comes once in a lifetime, yo
You better…
--  
Genre:Pop
Lyrics:I said, ooh, I'm blinded by the lights
No, I can't sleep until I feel your touch
I said, ooh, I'm drowning in the night
Oh, when I'm like this, you're the one I trust
--  
Genre:Country  
Lyrics:Country roads
Take me home
To the place I belong
West Virginia
Mountain mama
Take me home, country roads
--
"""

st.set_page_config(layout="centered", page_icon="🎵", page_title="Lyrics Generation Using Cohere")

st.header("Lyrics Generation")

model_type = "command"

with st.form("form"):
    genre_options = st.selectbox(
    'What genre would you like?',
    ('Rap', 'Pop', 'Country'))
    submitted = st.form_submit_button(label="Generate Lyrics!")

if submitted:
    with st.spinner("Writing something awesome ... Please wait a few seconds ... "):
        co = cohere.Client(api_key=CO_API_KEY)
        additional_prompt = f"""Genre:{genre_options}\nLyrics:"""
        if model_type == "generate":
            prompt = initial_prompt + additional_prompt
            gens = generate(co, prompt)
        else:
            prompt = f"Give the lyrics for a song."
            prompt += f" Make sure the genre is {genre_options}."
            gens = command(co, prompt)
        lyrics = gens[0].text
        st.markdown(f"## Lyrics for a song with genre {genre_options}")
        st.markdown(lyrics)

