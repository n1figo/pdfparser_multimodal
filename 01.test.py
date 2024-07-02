#  ! pip install -U langchain openai chromadb langchain-experimental # 최신 버전이 필요합니다 (멀티 모달을 위해)

import os
from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

# 파일 경로
fpath = "multi-modal/"
fname = "sample.pdf"


# PDF에서 요소 추출



import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# def extract_pdf_elements(path, fname):
#     """
#     PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
#     path: 이미지(.jpg)를 저장할 파일 경로
#     fname: 파일 이름
#     """
#     return partition_pdf(
#         filename=os.path.join(path, fname),
#         extract_images_in_pdf=True,  # PDF 내 이미지 추출 활성화
#         infer_table_structure=True,  # 테이블 구조 추론 활성화
#         chunking_strategy="by_title",  # 제목별로 텍스트 조각화
#         max_characters=4000,  # 최대 문자 수
#         new_after_n_chars=3800,  # 이 문자 수 이후에 새로운 조각 생성
#         combine_text_under_n_chars=2000,  # 이 문자 수 이하의 텍스트는 결합
#         image_output_dir_path=path,  # 이미지 출력 디렉토리 경로
#     )


# 요소를 유형별로 분류


def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출된 요소를 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements의 리스트
    """
    tables = []  # 테이블 저장 리스트
    texts = []  # 텍스트 저장 리스트
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))  # 테이블 요소 추가
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  # 텍스트 요소 추가
    return texts, tables






# 요소 추출
raw_pdf_elements = extract_pdf_elements(fpath, fname)

# 텍스트, 테이블 추출
texts, tables = categorize_elements(raw_pdf_elements)

# 선택사항: 텍스트에 대해 특정 토큰 크기 적용
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=4000, chunk_overlap=0  # 텍스트를 4000 토큰 크기로 분할, 중복 없음
)
joined_texts = " ".join(texts)  # 텍스트 결합
texts_4k_token = text_splitter.split_text(joined_texts)  # 분할 실행