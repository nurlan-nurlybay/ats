# 🤖 AI Recruiting Agent: Developer Test Assignment

### 🎯 Objective

Create a prototype of an AI agent that assists recruiters in processing resumes, matching them with job descriptions, and recommending the most suitable candidates.

---

### 📋 Input Data

* **Candidate Resumes**
* **Source:** Incoming emails sent to `cv@company.com`.
* **Format:** Any format chosen by the candidate (PDF, DOCX, etc.).
* **Requirement:** Automatic retrieval from the mailbox.


* **Job Descriptions (Vacancies)**
* **Format:** Stored as plain text.
* **Content:** Requirements, responsibilities, and tech stack.
* **Source:** Located in a separate folder or a database.



---

### ⚙️ Technical Requirements

#### 1. Email Integration

* Connect to the mailbox using **IMAP/SMTP** protocols.
* Automatically download new resume attachments.
* Save files to local storage/directory.

#### 2. Resume and Vacancy Processing

* Parse unstructured text into a **structured format**: skills, work experience, education.
* Apply **NLP approaches**: Tokenization, Embeddings, and Named Entity Recognition (NER).
* Perform keyword extraction.

#### 3. Matching Model

Implement **three distinct approaches** for comparison:

1. **Semantic Comparison:** Use Embeddings (Sentence-BERT / USE / HuggingFace) with **Cosine Similarity** as the metric.
2. **TF-IDF + ML Classifier:** Implement a baseline method to compare results against advanced models.
3. **LLM (Large Language Model):** Use OpenAI GPT, HuggingFace LLaMA, or Mistral. Create a **Prompt** that performs matching and provides a relevance explanation with a score from **0 to 1**.

#### 4. The Agent (Service)

* **Backend:** Develop the service using **FastAPI** or Flask.
* **Multilingualism:** Support for both **Russian and English**.
* **Functionality:** Accept a `job_id` or raw vacancy text; return the **Top 5** relevant candidates.
* **REST API:** Endpoint format `GET /recommendations?job_id=123`.
* **Frontend:** Create a simple UI using **Streamlit**.
* **Packaging:** Wrap the entire solution in **Docker**.

#### 5. Documentation

* Provide a **Jupyter Notebook** or a **Markdown report**.
* Include a description of the architecture and processing stages.
* Provide examples of the model's output and API usage.

---

### 🚀 Bonus Tasks

* Detailed explanations of the matching logic provided by the agent using an LLM.
* An automated update pipeline (automatically processing new data).

---

### 📊 Evaluation Criteria

| Criterion | Focus Area |
| --- | --- |
| **Resume Processing** | Correctness and robustness of parsing logic. |
| **Matching Quality** | Accuracy of candidate-to-vacancy mapping. |
| **Architecture** | Scalability and modularity of the solution. |
| **Code & Docs** | Cleanliness of code and clarity of documentation. |
| **Explainability** | How understandable the agent's decisions/scores are. |

---

### 💡 Quick Strategy Refresher for your "Vibe-coding":

* **For Requirement 1:** Use `imap_tools` (it's much cleaner than raw `imaplib`).
* **For Requirement 2:** Use `pdfplumber` for PDFs and `python-docx` for Word files.
* **For Requirement 3.1:** `sentence-transformers` is the standard library for SBERT.
* **For Requirement 4:** Use `docker-compose` to link the FastAPI backend and Streamlit frontend.
