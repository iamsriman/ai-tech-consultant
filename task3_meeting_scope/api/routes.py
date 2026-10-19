from fastapi import APIRouter, HTTPException
from task3_meeting_scope.api.schemas import MeetingRequest, DecisionRequest
from task3_meeting_scope.agents.orchestrator import Orchestrator
router=APIRouter(prefix="/api/v1")
service=Orchestrator()

@router.get("/health")
def health(): return {"status":"ok","projects":len(service.projects)}

@router.post("/projects/{project_id}/meetings")
def submit(project_id:str, request:MeetingRequest):
    try: return service.process(project_id,request.transcript,request.previous_version, provider_name=request.provider, model_name=request.model, knowledge_provider=request.knowledge_provider, knowledge_model=request.knowledge_model).model_dump()
    except Exception as exc: raise HTTPException(502,f"workflow failed: {type(exc).__name__}: {exc}")

@router.get("/projects/{project_id}")
def project(project_id:str):
    item=service.get(project_id)
    if not item: raise HTTPException(404,"project not found")
    return item.model_dump()

@router.get("/projects/{project_id}/versions/{version}")
def version(project_id:str,version:int):
    item=service.get(project_id)
    if not item or version<1 or version>len(item.versions): raise HTTPException(404,"version not found")
    return item.versions[version-1].model_dump()

@router.post("/projects/{project_id}/versions/{version}/approval")
def approval(project_id:str,version:int,request:DecisionRequest):
    try: return service.approve(project_id,version,request.approved).model_dump()
    except KeyError as exc: raise HTTPException(404,str(exc))
