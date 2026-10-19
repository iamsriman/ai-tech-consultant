from fastapi.testclient import TestClient
from pathlib import Path
from task3_meeting_scope.main import app
import task3_meeting_scope.api.routes as routes

def test_end_to_end_version_approval_and_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "service", __import__('task3_meeting_scope.agents.orchestrator',fromlist=['Orchestrator']).Orchestrator(tmp_path/'projects.json'))
    c=TestClient(app)
    first="The client wants a membership website with member search and email accounts. They need an administrator dashboard. Budget was not discussed."
    second="The client still wants member search and email accounts. They now require a custom mobile application and data migration."
    a=c.post('/api/v1/projects/p1/meetings',json={'transcript':first}).json()
    assert a['version']==1 and a['proposal']['status']=='pending_human_approval'
    assert c.post('/api/v1/projects/p1/versions/1/approval',json={'approved':True}).json()['proposal']['status']=='approved'
    b=c.post('/api/v1/projects/p1/meetings',json={'transcript':second,'previous_version':1}).json()
    assert b['version']==2 and b['change_summary']['added']
    assert c.get('/api/v1/projects/p1/versions/1').json()['proposal']['status']=='approved'

def test_validation_and_persistence(tmp_path, monkeypatch):
    from task3_meeting_scope.agents.orchestrator import Orchestrator
    store=tmp_path/'projects.json'; o=Orchestrator(store); item=o.process('p2','We need member search and an admin dashboard.')
    o2=Orchestrator(store); assert o2.get('p2').versions[0].version==item.version
    assert TestClient(app).post('/api/v1/projects/p2/meetings',json={'transcript':'short'}).status_code in (422,502)
