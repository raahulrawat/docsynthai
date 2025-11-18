<img src="assets/logo.png" width="80" />
DocSynthAI – Intelligent Document Processing MCP Server

DocSynthAI is an open-source Model Context Protocol (MCP) server designed to bring intelligence to unstructured documents.
It provides core IDP capabilities like document classification, extraction, validation, and canonical business rule enforcement — all accessible through a plug-and-play MCP interface.

🚀 What is DocSynthAI?

DocSynthAI transforms unstructured documents into structured, validated, enriched data.

It acts as a universal IDP engine that any LLM-powered or traditional application can plug into via MCP. Whether you’re dealing with invoices, forms, contracts, KYC docs, or free-flow text, DocSynthAI provides:

🧩 Classification

📄 Structured field extraction

🔍 OCR + NLP hybrid extraction

🧠 AI-based entity understanding

📏 Canonical business validations

🔒 Rule-based and ML-based QA checks

🔗 Multi-step pipelines

⚙️ Custom processors and plugins

🧶 Knowledge-graph–aligned output (optional)

✨ Key Features
🔹 1. Unstructured Document Classification

Identify document type (invoice, bank statement, contract, form, etc.)

AI + rule hybrid for high confidence

Multi-format support (PDF, JPG, PNG, TIFF, DOCX)

🔹 2. Extraction Pipeline

OCR with layout understanding

Table extraction

Key-value extraction

Multi-lingual text understanding

Intelligent page splitting

Multi-model & multimodal extraction

🔹 3. Validation Engine

Includes:

Structural validation

Mandatory field checks

Format + regex validations

Cross-field business rules

Canonicalization (convert names, dates, numbers to unified formats)

🔹 4. MCP Interface

DocSynthAI exposes itself as an MCP Server so any client (LLMs, agentic systems, automation tools) can:

Submit documents

Request extraction

Validate output

Ask for structured results

Retrieve logs & metadata

Perfect for integrating with:

ChatGPT Clients

Agentic platforms

Workflow orchestrators

Backend microservices

Data pipelines

🧱 Architecture Overview
          ┌───────────────────┐
          │   MCP Client      │
          └─────────┬─────────┘
                    │ Requests
          ┌─────────▼──────────┐
          │   DocSynthAI MCP   │
          │       Server       │
          └─────────┬──────────┘
      ┌──────────────┼────────────────┐
      │              │                │
┌─────▼─────┐ ┌─────▼─────┐ ┌────────▼────────┐
│ Classifier│ │ Extractor  │ │ Validation Core │
└───────────┘ └────────────┘ └─────────────────┘

🧪 Supported Document Types

Invoices

KYC documents

Utility bills

Bank statements

Insurance forms

Government IDs

Contracts

Custom templates

📦 Installation (Coming Soon)

Once published to PyPI:

pip install docsynthai


For local development:

git clone https://github.com/<your-org>/docsynthai.git
cd docsynthai
pip install -r requirements.txt


Start MCP server:

python -m docsynthai.server

🛠️ Usage as MCP Server

Your MCP client configuration:

{
  "servers": {
    "docsynthai": {
      "command": "python",
      "args": ["-m", "docsynthai.server"]
    }
  }
}

🎯 Roadmap

 Advanced table structure reconstruction

 Graph embedding + KG export

 LLM-assisted correction pipelines

 Plugin system for custom validators

 IDP pipeline visualizer

 Cloud deployment module

 Metrics dashboard
