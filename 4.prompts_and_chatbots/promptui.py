from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
load_dotenv()
import streamlit as st

model = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.7:cerebras",
    task="text-generation"
)

st.title("Chat with GLM-4.7")

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate(
             template="""
                      You are an expert AI researcher.

                     Explain the research paper: {paper}

                     Explanation Style: {style}

                         Explanation Length: {length}

                            Provide:
                            1. Main idea
                            2. Key contributions
                            3. How it works
                            4. Real-world applications
                            5. Limitations

                        Generate a well-structured explanation.
                        """,
                            input_variables=["paper", "style", "length"]
                        )

if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper':paper_input,
        'style':style_input,
        'length':length_input
    })
    st.write(result.content)
