# app/main.py
import re
import os
import torch
import smtplib
import torch.nn.functional as F
from email.message import EmailMessage
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.exa import ExaTools
from fpdf import FPDF

# Initialize FastAPI
app = FastAPI(title="Greenwashing Analyzer API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EXA_API_KEY=os.getenv("EXA_API_KEY")
PHI_API_KEY=os.getenv("PHI_API_KEY")

class AnalysisRequest(BaseModel):
    product_name: str
    email: EmailStr

# ------------------------------
# Core Analysis Functions
def extract_content(response):
    """Multi-stage content purification with metadata removal"""
    if not isinstance(response, str):
        response = str(response)

    # Remove technical artifacts
    response = re.sub(r'\b(content_type|event|messages|metrics|run_id|agent_id|session_id)\b=.*?(\s|$)', '', response)
    response = re.sub(r'\\Wn|\\n|\\', ' ', response)
    response = re.sub(r'\s+', ' ', response).strip()

    # Extract meaningful content
    content_match = re.search(r'content="([^"]+)"', response)
    clean_content = content_match.group(1) if content_match else response

    # Ensure complete sentences
    sentences = re.split(r'(?<=[.!?]) +', clean_content)
    if sentences:
        if not re.search(r'[.!?]$', sentences[-1]):
            sentences[-1] += '...'
        return ' '.join(sentences[:3])

    return clean_content[:300] + '...' if len(clean_content) > 300 else clean_content

def llm_preprocess_text(text):
    """Metadata-free text normalization"""
    cleaned = re.sub(r'<[^>]+>|{.*?}', '', text)
    cleaned = re.sub(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b', '', cleaned)

    summarizer = Agent(
        model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
        instructions="Extract key facts about sustainability and materials. Exclude technical specs, dates, and metrics.",
        tools=[ExaTools()],
    )
    return extract_content(summarizer.run(f"Clean and summarize: {cleaned}"))

def safe_run(agent, query, max_retries=3):
    """Error-resistant execution with query reformulation"""
    for attempt in range(max_retries):
        try:
            result = str(agent.run(query) or "")
            if "No data available" not in result:
                return result
        except Exception as e:
            if attempt == max_retries - 1:
                return "Information unavailable: Source query failed"

            # Reformulate query
            query = Agent(
                model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
                instructions="Rephrase this search query to be more effective",
            ).run(f"Improve query: {query}")

    return "Information unavailable: Maximum retries exceeded"

def evaluate_text(model_name, text, prompt):
    """Reliable scoring with explanation generation"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        inputs = tokenizer(
            f"{prompt}: {text[:1000]}",
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            score = round(probs[:, 1].item() * 100, 2)

    except Exception as e:
        print(f"⚠️ Model error: {str(e)}")
        score = 0.0

    # Generate explanation
    explainer = Agent(
        model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
        instructions="Explain this score in 1-2 complete sentences for a sustainability report",
    )
    explanation = extract_content(explainer.run(f"Score: {score}/100. Context: {text[:500]}"))

    return score, explanation

class GreenReportPDF(FPDF):
    """Formatted report generator"""
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 8, body)
        self.ln()

def generate_report(product_name, scores, final_score):
    """Create PDF output"""
    pdf = GreenReportPDF()
    pdf.title = f"Greenwashing Evaluation Report: {product_name}"
    pdf.add_page()

    # Scores Section
    pdf.chapter_title("Evaluation Metrics")
    for category, (score, explanation) in scores.items():
        pdf.chapter_body(
            f"{category} Score: {score}\n"
            f"Explanation: {explanation}\n"
        )

    # Final Score
    pdf.chapter_title("Overall Assessment")
    pdf.chapter_body(
        f"Final Green Score: {final_score['score']}/100\n"
        f"Calculation Methodology: {final_score['method']}\n"
        f"Recommendations: {final_score['recommendation']}"
    )

    filename = f"{product_name.replace(' ', '_')}_Greenwashing_Report.pdf"
    pdf.output(filename)
    return filename

# ------------------------------
# Email Service
def send_email(receiver_email: str, report_filename: str):
    try:
        msg = EmailMessage()
        msg["From"] = SMTP_USERNAME
        msg["To"] = receiver_email
        msg["Subject"] = "Greenwashing Analysis Report"
        
        msg.set_content(f"Please find attached your analysis report.\n\nBest regards,\nSustainability Team")

        with open(report_filename, "rb") as f:
            file_data = f.read()
            msg.add_attachment(
                file_data,
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(report_filename)
            )

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            
        return True
    except Exception as e:
        print(f"Email error: {str(e)}")
        return False

# ------------------------------
# API Endpoint
@app.post("/analyze")
async def analyze_product(request: AnalysisRequest):
    try:
        analysis_agent = Agent(
            model=Groq(
                id="mixtral-8x7b-32768",
                api_key=GROQ_API_KEY
            ),
            tools=[ExaTools(num_results=3)],
            instructions="Retrieve verifiable facts only. Cite sources. Exclude technical metadata.",
        )

        # Data Collection
        data_fields = {
            'sustainability': safe_run(analysis_agent, f"Official sustainability claims for {request.product_name}"),
            'certifications': safe_run(analysis_agent, f"Environmental certifications for {request.product_name}"),
            'transparency': safe_run(analysis_agent, f"Supply chain transparency reports for {request.product_name}"),
            'materials': safe_run(analysis_agent, f"Material composition analysis for {request.product_name}"),
        }

        # Data Processing
        processed_data = {k: llm_preprocess_text(v) for k, v in data_fields.items()}

        # Expert Evaluation
        scores = {
            'Claim Specificity': evaluate_text(
                "ESGBERT/EnvironmentalBERT-environmental",
                processed_data['sustainability'],
                "Assess claim specificity"
            ),
            'Certification Verification': evaluate_text(
                "ESGBERT/SocRoBERTa-social",
                processed_data['certifications'],
                "Evaluate certification credibility"
            ),
            'Supply Chain Transparency': evaluate_text(
                "climatebert/distilroberta-base-climate-sentiment",
                processed_data['transparency'],
                "Analyze transparency level"
            ),
            'Material Sustainability': evaluate_text(
                "ESGBERT/EnvironmentalBERT-action",
                processed_data['materials'],
                "Assess material sustainability"
            ),
        }

        # Final Scoring
        weights = {
            'Claim Specificity': 0.3,
            'Certification Verification': 0.2,
            'Supply Chain Transparency': 0.3,
            'Material Sustainability': 0.2
        }
        final_score_value = round(sum(scores[cat][0] * weight for cat, weight in weights.items()), 2)

        report_data = {
            'score': final_score_value,
            'method': "Weighted average (30% Claims, 20% Certifications, 30% Transparency, 20% Materials)",
            'recommendation': "Improve material sustainability disclosures and pursue recognized environmental certifications."
        }

        report_file = generate_report(request.product_name, scores, report_data)
        
        if not send_email(request.email, report_file):
            raise HTTPException(status_code=500, detail="Failed to send email")

        return {
            "status": "success",
            "message": "Report generated and sent successfully",
            "filename": report_file
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
