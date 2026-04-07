import os
import numpy as np
import cv2
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typhoon_ocr import ocr_document
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes

load_dotenv()

def load_data():
    DATAPATH = r"data/pdf"
    # print(f"Using data path: {DATAPATH}")
    # filenames= []
    # for filename in os.listdir(DATAPATH):
    #     if filename.endswith(".pdf"):
    #         file_path = os.path.join(DATAPATH, filename)
    #         filenames.append(file_path)
    # try: 
    #     with pdfplumber.open(filenames[0]) as pdf:
    #         print(pdf.pages[4].extract_text(layout=False))
            # pdf_text = []
            # for page in pdf.pages:
            #     text = page.extract_text()
            #     print(f"{text[:100]}...")  # Print the first 100 characters of the extracted text for debugging
            #     if text:
            #         pdf_text.append(text)
            # print(f"Extracted text from {filenames[0]} with {len(pdf_text)} pages.")
            # return pdf_text
    loader = PyPDFDirectoryLoader(DATAPATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {DATAPATH}")
    return documents
    # except Exception as e:
    #     print(f"Error occurred while loading data: {e}")
    # documents = []
    # # # pagenum = [32, 270, 62]
    # pagenum = [364]
    # for i, dir_name in enumerate(os.listdir(DATAPATH)):
    #     print(f"Processing {os.path.join(DATAPATH, dir_name)}...")# Limit to processing only the first document for testing
    #     texts = ocr_document(os.path.join(DATAPATH, dir_name), page_num=pagenum[i])
    #     for j, text in enumerate(texts):
    #         # print(f"Page {j+1} text: {text[:100]}...")  # Print the first 100 characters of each page's text
    #         documents.extend(texts)
    #     print(f"{documents}")
    return documents

def deskew(image):
    """
    improve the OCR accuracy by correcting the orientation of the text in the image.
    """
    if image is None or image.size == 0:
        return image

    if len(image.shape) == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

def OCR_load_data():
    DATAPATH = r"data/pdf"
    filename = os.listdir(DATAPATH)[0]  # Assuming there's only one PDF file in the directory
    file_path = os.path.join(DATAPATH, filename)
    with open(file_path, 'rb') as file:
        attachment_bytes = file.read()
    print(f"Loaded PDF file: {filename} with size {len(attachment_bytes)} bytes.")
    pages = convert_from_bytes(attachment_bytes)
    process_pages = []
    for page in pages:
        page_np = np.array(page)
        # PIL images are RGB; convert to BGR for OpenCV processing.
        if len(page_np.shape) == 3 and page_np.shape[2] == 3:
            page_np = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
        process_pages.append(deskew(page_np))
    print(f"Processed {len(process_pages)} pages for OCR.")
    ocr_texts = []
    print(f"Starting OCR on {len(process_pages)} pages...")
    for i, page in enumerate(process_pages):
        text = pytesseract.image_to_string(page, lang='tha')# Specify language if needed
        # print(f"Page {i+1} OCR text: {text[:100]}...")  # Print the first 100 characters of the OCR text for debugging
        ocr_texts.append(text)
    print(f"Extracted OCR text from {filename} with {len(ocr_texts)} pages.")
    return ocr_texts

def split_data(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text")
    # print(f"First text chunk: {texts[0].page_content}")
    return texts

def split_texts(texts) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    all_split_texts = []
    for text in texts:
        split_texts = text_splitter.split_text(text)
        all_split_texts.extend(split_texts) # append is used to add a single element to the list, while extend is used to add multiple elements from another list to the existing list.
    print(f"Split into {len(all_split_texts)} chunks of text")
    # print(f"First split text chunk: {all_split_texts[0]}")
    return all_split_texts

if __name__ == "__main__":
    # docs = load_data()
    # text = split_texts(docs)
    print("Starting OCR data loading...")
    ocr_texts = OCR_load_data()