# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.exa import ExaTools
from fpdf import FPDF

# Initialize FastAPI
app = FastAPI()

# Load environment variables (will be set in Render dashboard)
os.environ.update({
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
    "EXA_API_KEY": os.getenv("EXA_API_KEY"),
    # Add other API keys as needed
})

# Request model
class ProductRequest(BaseModel):
    product_name: str

# ----------- Existing Core Functions (Modified for API) -----------
# [Keep all the functions from your original code but:]
# 1. Remove the __main__ block
# 2. Modify generate_report to return PDF path
# 3. Add async/await where appropriate

@app.post("/generate-report")
async def generate_greenwashing_report(request: ProductRequest):
    try:
        product = request.product_name.strip()
        
        # Initialize components
        analysis_agent = Agent(
            model=Groq(id="mixtral-8x7b-32768"),
            tools=[ExaTools(num_results=3)],
            instructions="Retrieve verifiable facts only. Cite sources. Exclude technical metadata.",
        )
        
        # Data collection and processing
        data_fields = {
            'sustainability': safe_run(analysis_agent, f"Official sustainability claims for {product}"),
            'certifications': safe_run(analysis_agent, f"Environmental certifications for {product}"),
            'transparency': safe_run(analysis_agent, f"Supply chain transparency reports for {product}"),
            'materials': safe_run(analysis_agent, f"Material composition analysis for {product}"),
        }
        
        processed_data = {k: llm_preprocess_text(v) for k, v in data_fields.items()}
        
        # Scoring
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
        
        # Final score calculation
        weights = {
            'Claim Specificity': 0.3,
            'Certification Verification': 0.2,
            'Supply Chain Transparency': 0.3,
            'Material Sustainability': 0.2
        }
        final_score = round(sum(scores[cat][0] * weight for cat, weight in weights.items()), 2)
        
        # Generate report
        report_data = {
            'score': final_score,
            'method': "Weighted average (30% Claims, 20% Certifications, 30% Transparency, 20% Materials)",
            'recommendation': "Improve material sustainability disclosures and pursue recognized environmental certifications."
        }
        
        report_path = generate_report(product, scores, report_data)
        
        return FileResponse(
            report_path,
            media_type='application/pdf',
            filename=report_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))