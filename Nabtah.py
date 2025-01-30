import streamlit as st
import openai
from Matcher import Analyze

st.set_page_config(page_title="Nabtah", page_icon="🌿", layout="centered")

st.logo("Images/Medvation_logo.png", link='https://www.medvation.net')
st.html(""" <style> [alt=Logo] { height: 4rem; } </style> """) # logo size

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"]) # Opeai API Key
# LLM Model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o"

# Chat history
if "texts" not in st.session_state:
    st.session_state.texts = []


if "page" not in st.session_state: # we have two pages. Chat page and Recommendation page
    st.session_state.page = "Chat" 

if st.session_state.page == "Chat": # Chat Page

    # title
    st.markdown("<h1 style='text-align: center; color: #A8D4B4;'>Nabtah 🌿</h1>", unsafe_allow_html=True) 

    # Initial system message
    if not st.session_state.texts:
        st.session_state.texts.append({
            "role": "system",
            "content": """
                        You are a friendly chatbot engaging in a casual conversation with a teenager. Your goal is to naturally guide the conversation towards their daily life, interests, and personal preferences in a way that feels effortless and organic, while subtly exploring aspects of the Holland Codes (RIASEC) without making it obvious.

                        Conversation Guidelines:
                        1. Start with a friendly greeting – Ask how their day is going in a warm and casual way.
                        2. Encourage them to talk about recent activities – Ask about fun things they've been doing, shows they've watched, or hobbies they've enjoyed.
                        3. Casually introduce school subjects – Inquire about subjects they find interesting or enjoyable, without focusing on grades.
                        4. Explore their free time activities and hobbies – Encourage them to share how they spend their spare time, such as sports, arts, gaming, or volunteering.
                        5. Subtly inquire into their Holland Code traits:
                        - Realistic (R): "Do you enjoy hands-on activities like building, fixing, or exploring outdoors?"
                        - Investigative (I): "Have you found yourself curious about how things work or why things happen?"
                        - Artistic (A): "Do you like expressing yourself through art, music, or writing?"
                        - Social (S): "Do you enjoy helping others or working in a team?"
                        - Enterprising (E): "Do you like leading projects or convincing people of new ideas?"
                        - Conventional (C): "Do you find organizing tasks or planning events satisfying?"
                        6. Prompt them about future interests – Ask if there's anything they’re excited to learn or try, ensuring it doesn’t feel like a career discussion.
                        7. Understand their preferred work style – Find out if they enjoy working alone, in groups, or through hands-on projects by asking about recent activities they enjoyed.

                        Tone & Restrictions:
                        - Keep the conversation fun, light, and pressure-free.
                        - Maintain a friendly, conversational tone, making the teen feel comfortable and understood.

                        Your role is to engage in an enjoyable conversation while subtly understanding their preferences aligned with the Holland Codes, without tracking or analyzing their responses.
                        """
        })

    # Display History on each rerun
    for text in st.session_state.texts[1:]:
        with st.chat_message(text["role"]):
            st.markdown(text["content"])


    user_prompt = st.chat_input("How is your day going?")
    if user_prompt:
        
        with st.chat_message("user"): 
            st.markdown(user_prompt) # Display User Input   
        st.session_state.texts.append({"role": "user", "content": user_prompt}) # add user input to history

        # Assistant (AI Model):  
        with st.chat_message("assistant"):
            model_respone =  client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages = [{"role": m["role"], "content":m["content"]} for m in st.session_state.texts],
                stream=True
            )
            response = st.write_stream(model_respone) # Display mode response in stream form
        st.session_state.texts.append({"role":"assistant","content":response}) # add model response to history

    with st.sidebar:

        st.sidebar.markdown("### View Your Career Recommendation")
        st.sidebar.markdown("Click the button below to see your recommended career track!")
        # Button to switch to the Recommendations page
        if st.button("Recommendation"):
            st.session_state.page = "Recommendation"




elif st.session_state.page == "Recommendation": # Recommendations page
    
    # collect user inputs from the conversation
    user_inputs = []
    for txt in st.session_state.texts:
        if txt['role'] == 'user':
            user_inputs.append(txt['content'])
    user_input = ". ".join(user_inputs)


    thx_msg =  client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages = st.session_state.texts + [{"role": "system", "content": "Please provide a positive and encouraging message to the teen. It will be used in the final reccomendation report."}],
    )

    # Displaying options based on their interests and hobbies
    actions =  client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages = st.session_state.texts + [{"role": "system", "content": "Please Provide suggestions for next steps, like what the teen can explore further or ways to develop their skills in a friendly way. It will be used in the final reccomendation report."}],
    )

    thx_msg = thx_msg.choices[0].message.content
    actions = actions.choices[0].message.content


    analyzeConv = Analyze() 
    result = analyzeConv.analyze(user_input) # Analyzing the conversation
    if result['status']=='warning': # Display a warning messgae if the assesment is not completed
        st.warning(result['message'])
        st.stop()
    result = result['message']


    st.title("Here's the Results 🥳") # Recommendation page title

    st.write(thx_msg)

    original_fig = analyzeConv.radarChart(original=True) # Radar chart using the original scores
    st.plotly_chart(original_fig)

    RIASEC_fig = analyzeConv.radarChart(RIASEC=True) # Radar chart using the RIASEC scores
    st.plotly_chart(RIASEC_fig)

    total_fig = analyzeConv.radarChart() # Radar chart using the original+RIASEC scores
    st.plotly_chart(total_fig)

    # Recommended track
    st.markdown(f"<span style='color:#6b9080; font-weight:bold; font-size:25px'>{result}</span>", unsafe_allow_html=True) 

    st.write(actions)



    
   
