def compare(old, new):
    old_ids={r.id:r.statement for r in old.analysis.requested_features}; new_ids={r.id:r.statement for r in new.analysis.requested_features}
    same=[v for k,v in new_ids.items() if k in old_ids and old_ids[k]==v]; added=[v for k,v in new_ids.items() if k not in old_ids]; removed=[v for k,v in old_ids.items() if k not in new_ids]
    old_hours=sum(t.estimated_hours for t in old.scope.tasks); new_hours=sum(t.estimated_hours for t in new.scope.tasks)
    return {"unchanged":same,"added":added,"removed":removed,"outside_original_scope":added,"old_hours":old_hours,"new_hours":new_hours,"hours_delta":new_hours-old_hours,"questions":new.analysis.unanswered_questions}
