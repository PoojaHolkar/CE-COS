import os
from dotenv import load_dotenv
import json
import io
import sys
import ibm_boto3
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from ibm_botocore.client import Config
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.document import TableItem
from langchain_core.documents import Document
from docling_core.types.doc.labels import DocItemLabel
from docling.datamodel.base_models import DocumentStream
import requests

# Load environment variables from key.env
load_dotenv("key.env")

app = FastAPI()

# Watsonx config
config = {
    "url": f"{os.getenv('WATSONX_URL')}/ml/v1/text/chat?version=2023-05-29",
    "model_id": "meta-llama/llama-3-2-11b-vision-instruct",
    "project_id": os.getenv('PROJECT_ID'),
    "max_tokens": 300,
    "time_limit": 10000
}

# IBM COS config
cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=os.getenv("API_KEY"),
    ibm_service_instance_id=os.getenv("SERVICE_INSTANCE_ID"),
    config=Config(signature_version="oauth"),
    endpoint_url=os.getenv("COS_ENDPOINT"),
)

bucket_name = "sharecat-upload"
write_bucket = "sharecat-write"

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

pdf_pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    generate_picture_images=True
)
format_options = {
    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
}
converter = DocumentConverter(format_options=format_options)


@app.get("/process")
def process_files():
    conversions = {}
    pdf_keys = [obj["Key"] for obj in cos.list_objects_v2(Bucket=bucket_name)["Contents"] if obj["Key"].endswith(".pdf")]
    print(f"DEBUG: Found {len(pdf_keys)} PDF files")
    print("DEBUG: PDF Keys =>", pdf_keys)

    for key in pdf_keys:
        print(f"Processing: {key}")
        obj = cos.get_object(Bucket=bucket_name, Key=key)
        file_bytes = io.BytesIO(obj["Body"].read())

        doc_stream = DocumentStream(
            name=key,
            stream=file_bytes,
            mime_type="application/pdf"
        )
        conversions[key] = converter.convert(source=doc_stream).document

    response = cos.list_objects_v2(Bucket=bucket_name)
    print("DEBUG: COS Response Keys =>", [obj["Key"] for obj in response.get("Contents", [])])

    doc_id = 0
    texts: list[Document] = []
    for source, docling_document in conversions.items():
        for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(docling_document):
            items = chunk.meta.doc_items
            if len(items) == 1 and isinstance(items[0], TableItem):
                continue
            refs = " ".join(map(lambda item: item.get_ref().cref, items))
            text = chunk.text
            document = Document(
                page_content=text,
                metadata={
                    "doc_id": (doc_id := doc_id + 1),
                    "source": source,
                    "ref": refs,
                },
            )
            texts.append(document)

    doc_id = len(texts)
    tables: list[Document] = []
    for source, docling_document in conversions.items():
        for table in docling_document.tables:
            if table.label in [DocItemLabel.TABLE]:
                ref = table.get_ref().cref
                text = table.export_to_markdown()
                document = Document(
                    page_content=text,
                    metadata={
                        "doc_id": (doc_id := doc_id + 1),
                        "source": source,
                        "ref": ref
                    },
                )
                tables.append(document)

    for source, docling_document in conversions.items():
        extracted_data = {
            "text": "\n\n".join(doc.page_content for doc in texts if doc.metadata["source"] == source),
            "tables": [doc.page_content for doc in tables if doc.metadata["source"] == source],
        }

        json_filename = f"{source.replace('/', '_').replace('.pdf', '')}_extracted.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2)

        print(f"Generated JSON for {source}: {json_filename}")

    def get_access_token():
        iam_url = 'https://iam.cloud.ibm.com/identity/token'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'urn:ibm:params:oauth:grant-type:apikey',
            'apikey': os.environ['API_KEY']
        }
        response = requests.post(iam_url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json().get('access_token')
        else:
            print(f"Failed to get access token: {response.status_code}")
            return None

    def invoke_wx_ai(config, prompt):
        headers = {
            "Authorization": f"Bearer {get_access_token()}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        body = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant specializing in extracting financial details from Form 16 documents. Your task is to analyze the provided text and table data and return accurate structured data."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model_id": config["model_id"],
            "project_id": config["project_id"],
            "decoding_method": "greedy",
            "repetition_penalty": 1,
            "max_tokens": 1000
        }

        response = requests.post(config["url"], headers=headers, json=body)
        if response.status_code == 200:
            return {"response": response.json()['choices'][0]['message']['content']}
        else:
            return {"error": response.text}

    output_dir = "extracted_json_files"
    os.makedirs(output_dir, exist_ok=True)

    for file in pdf_keys:
        print(f"Processing {file}...")
        json_filename = f"{file.replace('/', '_').replace('.pdf', '')}_extracted.json"
        with open(json_filename, "r") as f:
            extracted_data = json.load(f)

        prompt = f"""
        This is a Form 16 document containing salary and tax deduction details.

        Extracted Text:
        {extracted_data['text']}

        Extracted Tables:
        {json.dumps(extracted_data['tables'], indent=2)}

        You are a tax assistant. From the extracted text and tables, extract and return the following fields in **flat JSON format**. All values must be numeric (no text, no calculations).

        ⚠️ Strict Instructions:
        - Only return valid JSON.
        - No explanations, no extra comments.
        - If a value is not available, use 0.0.
        - DO NOT use formulas like 50000 + 2400 — instead return the summed value (e.g., 52400).

        Expected JSON structure:
        {{
        "gross_salary": <Total gross salary from Section 17>,
        "exemptions_section_10": <Total exemptions under Section 10 like HRA>,
        "allowances": <Sum of standard deduction, entertainment allowance, professional tax>,
        "deductions_chapter_via": <Total deductions under Chapter VI-A>,
        "total_income": <Total taxable income>,
        "tax_on_total_income": <Tax calculated on total taxable income>,
        "rebate_87a": <Rebate if any>,
        "cession_or_surcharge": <Cess or surcharge total>,
        "relief_section_89": <Relief under Section 89 if any>,
        "net_tax_payable": <Tax payable after relief>,
        "total_other_income": <Other income if reported>
        }}

        User Query: Please extract all tax-relevant financial fields from this Form 16.
        """
        response = invoke_wx_ai(config, prompt)
        print(f"AI Response for {file}: {response['response']}")

        try:
            parsed_inner_json = json.loads(response["response"])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {file}: {response['response']}")
            continue

        final_output = {
            "gross_salary": parsed_inner_json["gross_salary"],
            "exemptions_section_10": parsed_inner_json["exemptions_section_10"],
            "allowances": parsed_inner_json["allowances"],
            "deductions_chapter_via": parsed_inner_json["deductions_chapter_via"],
            "total_income": parsed_inner_json["total_income"],
            "tax_on_total_income": parsed_inner_json["tax_on_total_income"],
            "rebate_87a": parsed_inner_json["rebate_87a"],
            "cession_or_surcharge": parsed_inner_json.get("health_cess_or_education", 0.0),
            "relief_section_89": parsed_inner_json["relief_section_89"],
            "net_tax_payable": parsed_inner_json["net_tax_payable"],
            "total_other_income": parsed_inner_json["total_other_income"],
            "tax_deducted": round(parsed_inner_json["net_tax_payable"]),
            "total_amount_credited": parsed_inner_json["gross_salary"]
        }

        final_json_str = json.dumps(final_output, indent=2)
        cos_key = f"{file.replace('/', '_').replace('.pdf', '')}_output.json"

        cos.put_object(
            Bucket=write_bucket,
            Key=cos_key,
            Body=final_json_str.encode("utf-8"),
            ContentType='application/json'
        )
        print(f"✅ Uploaded {cos_key} to bucket {write_bucket}")
        os.remove(json_filename)

    return JSONResponse(content={
        "status": "success",
        "message": f"Processed {len(pdf_keys)} files and uploaded JSONs to '{write_bucket}'..."
    })