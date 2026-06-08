import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import app as agent_app

api_app = FastAPI()

# CORS setup
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    candidate_id: str
    internship_id: str

@api_app.post("/analyze")
async def run_analysis(request: AnalysisRequest):
    try:
        initial_state = {
            "candidate_id": request.candidate_id,
            "internship_id": request.internship_id,
            "analysis_report_path": f"reports/{request.candidate_id}_report.json"
        }
        result = agent_app.invoke(initial_state)
        return {"status": "success", "report_path": result["analysis_report_path"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New Endpoint: Fetch the generated report
@api_app.get("/report/{candidate_id}")
async def get_report(candidate_id: str):
    file_path = f"reports/{candidate_id}_report.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    else:
        raise HTTPException(status_code=404, detail="Report not found")