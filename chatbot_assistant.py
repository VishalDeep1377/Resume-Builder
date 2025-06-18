import streamlit as st
import openai

def chat_with_resume_assistant(resume_text):
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": (
                "You are a friendly, expert career coach and resume writer. "
                "You help users improve their resumes, explain mistakes, and give job-hunting advice."
            )}
        ]

    st.header("🤖 Ask Resume Assistant")
    st.markdown("Ask anything about your resume, job search, or career!")

    # Display chat history
    for msg in st.session_state['messages']:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg['content'])
        elif msg['role'] == 'assistant':
            st.chat_message("assistant").write(msg['content'])

    # User input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        st.session_state['messages'].append({"role": "user", "content": user_input})

        # Compose prompt with resume context
        resume_context = f"Here's the user's resume text: {resume_text}\n"
        full_prompt = resume_context + user_input

        # Call OpenAI API
        openai_api_key = st.session_state.get('openai_api_key')
        if openai_api_key:
            client = openai.OpenAI(api_key=openai_api_key)
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        st.session_state['messages'][0],  # system prompt
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.7,
                )
                answer = response.choices[0].message.content.strip()
        else:
            answer = "Please enter your OpenAI API key in the sidebar to use the assistant."

        st.session_state['messages'].append({"role": "assistant", "content": answer})
        st.rerun() 