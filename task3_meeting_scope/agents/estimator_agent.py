from task3_meeting_scope.models import DeveloperTask, Requirement, FeatureCheck
class EstimatorAgent:
    def estimate(self, requirements, checks):
        checked={x.requirement_id:x for x in checks}; tasks=[]
        for i,req in enumerate(requirements,1):
            check=checked.get(req.id); hours=4.0 if check and check.supported else 12.0
            tasks.append(DeveloperTask(task_id=f"TASK-{i}",description=f"Configure or implement {req.statement}",expected_result=f"Verified {req.statement} workflow",estimated_hours=hours,assumptions=["Estimate excludes unconfirmed integrations and content migration"]))
        return tasks
