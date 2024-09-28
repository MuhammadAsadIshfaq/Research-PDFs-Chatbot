import os
import google.generativeai as genai
import warnings
from fpdf import FPDF
import gradio as gr
import re

from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore")

# Set your Google API key as an environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API')
if GOOGLE_API_KEY is None:
    raise ValueError("Google API key is not set. Make sure it's defined in your Hugging Face secret variables.")
# Configure the Google Generative AI library
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model and embeddings
model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY, temperature=0.2, convert_system_message_to_human=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Define function to process PDF and generate answers
def process_pdf_and_qa(pdf_file, question):
    # Load and split PDF
    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":6})

    # Set up QA chain
    template = """Use the following pieces of context to answer the question at the end. 
                  If the answer to the question includes equations, make sure to display the equations properly. 
                  Also, if relevant, explain the meaning of the equation and any key terminology.
                  If the question is beyond the context, generate an answer using general knowledge and mention that.
                  Always say "thanks for asking!" at the end.
                  
                  Context:
                  {context}
                  Question: {question}
                  Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    # Get the answer
    result = qa_chain({"query": question})
    return result["result"]

# Define function to create PDF
from fpdf import FPDF
import os

# Define function to create PDF with Unicode support
def create_pdf(question, answer):
    pdf = FPDF()
    pdf.add_page()

    # Make sure the font file path is correct or uploaded properly
    font_path = "DejaVuSans.ttf"  # Change this if the path is different

    # Add a Unicode-compliant font (make sure the font is in the correct directory)
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
    else:
        raise RuntimeError(f"TTF Font file not found at: {font_path}")
    
    # Set the font to DejaVu (Unicode-supporting font)
    pdf.set_font("DejaVu", size=12)
    
    # Add content
    pdf.cell(200, 10, txt="Question:", ln=True)
    pdf.multi_cell(0, 10, txt=question)
    pdf.cell(200, 10, txt="Answer:", ln=True)
    pdf.multi_cell(0, 10, txt=answer)
    
    # Output to file
    pdf_output = "/tmp/answer.pdf"
    pdf.output(pdf_output)
    return pdf_output

    
# Define Gradio interface
def gradio_interface(pdf_file, question):
    answer = process_pdf_and_qa(pdf_file, question)
    pdf_path = create_pdf(question, answer)
    return answer, pdf_path

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a Question")],
    outputs=[gr.Textbox(label="Answer"), gr.File(label="Download Answer as PDF")],
    title="Research Bot",
    description="Upload a PDF file and ask questions related to its content. Download the answer as a PDF."
)

if __name__ == "__main__":
    iface.launch(share=True)  # Set `share=True` to create a public link
