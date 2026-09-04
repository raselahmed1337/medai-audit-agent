"""Benchmark tasks with gold answers derived from the fixture corpus.

Each task: a research query, the gold topic (defines gold citations via
corpus.gold_relevant), and a gold pooled effect computed independently by the
evaluator from the gold-relevant papers (not from agent internals).
"""
from __future__ import annotations

from medai.corpus import gold_relevant
from medai.tools.analysis import meta_analysis_fixed
from medai.tools.extraction import extract_effects
from medai.corpus import corpus_by_id

TASKS = [
    dict(task_id="T1", query="aspirin myocardial infarction risk", topic="aspirin_mi"),
    dict(task_id="T2", query="aspirin primary prevention myocardial infarction", topic="aspirin_mi"),
    dict(task_id="T3", query="low dose aspirin cardiovascular events", topic="aspirin_mi"),
    dict(task_id="T4", query="statin LDL cholesterol reduction", topic="statin_ldl"),
    dict(task_id="T5", query="atorvastatin rosuvastatin LDL lowering", topic="statin_ldl"),
    dict(task_id="T6", query="statin therapy hypercholesterolemia randomized", topic="statin_ldl"),
    dict(task_id="T7", query="SSRI depression response odds", topic="ssri_depression"),
    dict(task_id="T8", query="sertraline fluoxetine major depressive disorder", topic="ssri_depression"),
    dict(task_id="T9", query="antidepressant SSRI elderly depression", topic="ssri_depression"),
    dict(task_id="T10", query="aspirin MI pooled effect estimate", topic="aspirin_mi"),
    dict(task_id="T11", query="ezetimibe statin combination LDL", topic="statin_ldl"),
    dict(task_id="T12", query="escitalopram paroxetine depression trial", topic="ssri_depression"),
]


def gold_answer(task: dict) -> dict:
    """Gold citations + gold pooled estimate from gold-relevant papers only."""
    ids = gold_relevant(task["topic"])
    by_id = corpus_by_id()
    effects = [e for pid in ids for e in extract_effects(by_id[pid])]
    analysis = meta_analysis_fixed(effects)
    return dict(gold_citations=ids, gold_pooled=analysis["pooled_point"],
                gold_ci=(analysis["ci_lo"], analysis["ci_hi"]))
