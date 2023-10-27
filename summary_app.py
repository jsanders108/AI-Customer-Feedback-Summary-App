#Importing the necessary LangChain libraries
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Loading environment variables
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#importing streamlit
import streamlit as st

#Reading the text of the emails into the "text" variable
def load_document(file):
    with open(file, encoding='utf-8') as f:
        text = f.read()
    return text

#Splitting the text into chunks
def chunk_data(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents([text])
    print(len(chunks))
    return chunks

def summarize_feedback(data, chunks):
    #Creating the LLM
    llm = ChatOpenAI(openai_api_key = os.environ.get('OPENAI_API_KEY'), model_name='gpt-3.5-turbo', temperature=0)
    
    #Refining with custom prompts
    text = data
    prompt_template = """
        Write a concise summary of the following, extracting the key information:
        Text: {text}
        Concise Summary:
    """
    initial_prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

    refine_template = """
        Your job is to produce a final summary of customer feedback emails.
        I have provided an existing summary up to a certain point: {existing_answer}
        Please refine the existing summary with some more context below
        ------------------------------------------------------------------
        {text}
        ------------------------------------------------------------------
        Please provide an Executive Summary. The Executive Summary
        should highlight both the positive and negative aspects of the product. 
        Please split the summary into relevant paragraphs for good readability. 
    """
    refine_prompt = PromptTemplate(template=refine_template, input_variables=["existing_answer", "text"])


    #Defining the chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type='refine',
        question_prompt=initial_prompt,
        refine_prompt=refine_prompt,
        verbose=True
    )

    #Running the chain
    output_summary = chain.run(chunks)

    return output_summary


#Creating the streamlit user interface
if __name__ == "__main__":

    #Creating the header and subheader on the page
    st.image('img.png')
    st.subheader('Customer Feedback Summary Application')

    #Allowing the user to upload any text file and select the chunk size
    uploaded_file = st.file_uploader('Upload a file:', type=['txt'])
    chunk_size = st.number_input('Chunk size:', min_value=100, max_value=6000, value=4000)

    #Creating a "Summarize Feedback" button
    prep_data = st.button('Summarize Feedback')

    #If a text file has been uploaded and the "Summarize Feedback" button has been clicked,
    #then running the chunk_data and summarize_feedback functions
    if uploaded_file and prep_data:
        with st.spinner('Summarizing the feedback ... '):

            #Writing the uploaded file to a new file in the local folder
            bytes_data = uploaded_file.read()
            file_name = os.path.join('./', uploaded_file.name)
            with open(file_name, 'wb') as f:
                f.write(bytes_data) 
            data = load_document(file_name)

            #Running the chunk_data function, passing in the text from the uploaded file
            chunks = chunk_data(data, chunk_size, 20)

            #Running the summarize_feedback function, passing in the newly created chunks 
            summary = summarize_feedback(data, chunks)
            st.text_area("Summary:", value=summary)

            
    
    

