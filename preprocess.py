import os
import glob
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Tesseract path (update for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def is_pdf_image_based(pdf_path, check_pages=1):
    """Check if PDF is image-based by trying to extract text"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Check first 'check_pages' pages
    for doc in docs[:check_pages]:
        text = doc.page_content.strip()
        if len(text) > 50:  # If decent text is found, it's text-based
            return False
    return True  # Otherwise assume image-based

def extract_text_from_image_pdf(pdf_path):
    """Extract text from image-based PDF using OCR"""
    try:
        images = convert_from_path(pdf_path, dpi=300)
        full_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            full_text += f"Page {i+1}:\n{text}\n\n"
        return full_text.strip() if full_text.strip() else None
    except Exception as e:
        print(f"⚠️ OCR Error for {os.path.basename(pdf_path)}: {str(e)}")
        return None

def load_subject_documents(subject_path):
    """Load documents from a subject folder, handling PDFs and PPTs"""
    documents = []

    # Process all PDFs
    for pdf_file in glob.glob(os.path.join(subject_path, "*.pdf")):
        print(f"🔍 Checking PDF: {os.path.basename(pdf_file)}")
        try:
            if is_pdf_image_based(pdf_file):
                print(f"🖼️ Detected image-based PDF: {os.path.basename(pdf_file)}")
                ocr_text = extract_text_from_image_pdf(pdf_file)
                if ocr_text:
                    documents.append(Document(
                        page_content=ocr_text,
                        metadata={"source": pdf_file, "type": "ocr_pdf"}
                    ))
                    print(f"✅ OCR extracted text: {os.path.basename(pdf_file)}")
                else:
                    print(f"❌ OCR failed: {os.path.basename(pdf_file)}")
            else:
                print(f"📄 Detected text-based PDF: {os.path.basename(pdf_file)}")
                loader = PyPDFLoader(pdf_file)
                docs = loader.load_and_split()
                documents.extend(docs)
                print(f"✅ Loaded text-based PDF: {os.path.basename(pdf_file)} (pages: {len(docs)})")
        except Exception as e:
            print(f"❌ Failed PDF: {os.path.basename(pdf_file)} - {str(e)}")

    # Process all PPT/PPTX files
    for ppt_file in glob.glob(os.path.join(subject_path, "*.ppt*")):
        try:
            loader = UnstructuredPowerPointLoader(ppt_file)
            docs = loader.load()
            documents.extend(docs)
            print(f"✅ Loaded PPT: {os.path.basename(ppt_file)}")
        except Exception as e:
            print(f"❌ Failed PPT: {os.path.basename(ppt_file)} - {str(e)}")
    
    return documents

def process_subjects(base_folder):
    """Main processing function"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    subjects = os.listdir(base_folder)  # Automatically detect all subject folders
    
    for subject in subjects:
        subject_path = os.path.join(base_folder, subject)
        print(f"\n{'='*40}\n📂 Processing {subject.upper()}...\n{'='*40}")
        
        if not os.path.isdir(subject_path):
            print(f"⚠️ Skipping non-folder: {subject_path}")
            continue
        
        documents = load_subject_documents(subject_path)
        if not documents:
            print(f"⚠️ No valid documents in {subject}")
            continue
        
        chunks = text_splitter.split_documents(documents)
        print(f"✂️ Created {len(chunks)} chunks")
        
        if chunks:
            os.makedirs("faiss_index", exist_ok=True)
            try:
                vector_db = FAISS.from_documents(chunks, embedding_model)
                save_path = f"faiss_index/{subject}_faiss"
                vector_db.save_local(save_path)
                print(f"💾 Saved FAISS index to {save_path}")
            except Exception as e:
                print(f"❌ Failed to save index: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting document processing...")
    process_subjects("data_base_pdf")
    print("\n🎉 Processing complete!")
