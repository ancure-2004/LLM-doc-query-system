import os
import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def ingest_documents(data_dir="data"):
    all_chunks = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)

        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "doc_name": filename,
                "chunk_id": i,
                "text": chunk
            })

    return all_chunks

if __name__ == "__main__":
    chunks = ingest_documents()
    print(f"✅ Extracted {len(chunks)} chunks.")
    print(f"📄 Example Chunk:\n{chunks[0]}")
