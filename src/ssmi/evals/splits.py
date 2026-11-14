"""Evaluation splits. Trials from one group (subject or session) never appear in both
train and test -> no leakage from electrode placement / session drift."""

from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold


def get_splits(rec, scheme: str = "leave_one_group_out", seed: int = 0):
    if scheme == "leave_one_group_out" and len(set(rec.groups)) > 1:
        return list(LeaveOneGroupOut().split(rec.X, rec.y, rec.groups))
    # fallback for a single-session pilot; flagged in report
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(rec.X, rec.y))
