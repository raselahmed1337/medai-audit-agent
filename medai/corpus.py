"""Offline fixture corpus of medical abstracts with gold relevance labels.

Used as the default (reproducible) retrieval source; PubMed is only queried
when the caller explicitly passes live=True (see tools/retrieval.py).

Each paper: id, topic (gold relevance key or None for distractors), title,
abstract. Effect sentences follow the pattern the extractor detects.
"""

FIXTURE_CORPUS = [
    # --- topic: aspirin_mI ---
    dict(id="P001", topic="aspirin_mi", title="Aspirin and myocardial infarction: a randomized trial of 20,000 physicians",
         abstract="We randomized 20,000 male physicians to aspirin 325 mg or placebo. The primary endpoint was myocardial infarction. Aspirin reduced the risk of myocardial infarction (risk ratio 0.56, 95% CI 0.45 to 0.70). Bleeding events were slightly increased but the benefit dominated. We conclude aspirin is effective for primary prevention of MI in this population."),
    dict(id="P002", topic="aspirin_mi", title="Low-dose aspirin for secondary prevention of myocardial infarction",
         abstract="In 4,500 patients with prior MI, low-dose aspirin (75 mg) was compared with placebo. Recurrent myocardial infarction occurred less often with aspirin (risk ratio 0.68, 95% CI 0.54 to 0.86). Gastroduodenal bleeding was more frequent in the aspirin arm. Low-dose aspirin is recommended for secondary prevention."),
    dict(id="P003", topic="aspirin_mi", title="Aspirin in elderly patients: cardiovascular outcomes trial",
         abstract="We studied 15,000 patients aged 70+ randomly assigned to enteric-coated aspirin 100 mg or placebo. The composite of myocardial infarction, stroke or cardiovascular death showed risk ratio 0.91, 95% CI 0.79 to 1.05, a non-significant reduction. Major hemorrhage was significantly increased with aspirin. In the elderly, aspirin did not significantly reduce cardiovascular events."),
    dict(id="P004", topic="aspirin_mi", title="Meta-analytic overview of aspirin in primary prevention",
         abstract="This pooled analysis of individual data examined aspirin versus control for first cardiovascular events. Myocardial infarction was reduced with aspirin (risk ratio 0.79, 95% CI 0.71 to 0.88) at the cost of major bleeding. Findings support individualized decisions on aspirin for primary prevention."),
    # --- topic: statin_ldl ---
    dict(id="P010", topic="statin_ldl", title="Atorvastatin 10 mg versus placebo for LDL-cholesterol reduction",
         abstract="A total of 800 hypercholesterolemic adults received atorvastatin 10 mg or placebo for 12 weeks. Mean LDL-cholesterol fell by 1.9 mmol/L more with atorvastatin (mean difference 1.9, 95% CI 1.7 to 2.1). The drug was well tolerated. Atorvastatin potently lowers LDL-cholesterol."),
    dict(id="P011", topic="statin_ldl", title="Rosuvastatin versus placebo: LDL reduction in a 24-week trial",
         abstract="We randomly assigned 1,200 patients to rosuvastatin 20 mg or placebo. The between-group difference in LDL-cholesterol at 24 weeks was mean difference 2.4, 95% CI 2.2 to 2.6 mmol/L, favouring rosuvastatin. Adverse events were balanced between arms. Rosuvastatin produced large LDL-cholesterol reductions."),
    dict(id="P012", topic="statin_ldl", title="Simvastatin 40 mg for hypercholesterolaemia: randomized double-blind trial",
         abstract="In this double-blind trial of 900 participants, simvastatin 40 mg reduced LDL-cholesterol relative to placebo (mean difference 1.5, 95% CI 1.3 to 1.7 mmol/L). Rates of myalgia were low and similar between groups. Simvastatin is an effective LDL-lowering agent."),
    dict(id="P013", topic="statin_ldl", title="Ezetimibe added to statin therapy: effect on LDL-cholesterol",
         abstract="Adding ezetimibe 10 mg to baseline statin therapy in 1,000 patients produced an additional LDL-cholesterol reduction (mean difference 0.4, 95% CI 0.3 to 0.5 mmol/L versus placebo add-on). The combination was well tolerated. Ezetimibe provides incremental LDL lowering."),
    # --- topic: ssri_depression ---
    dict(id="P020", topic="ssri_depression", title="Sertraline versus placebo in major depressive disorder",
         abstract="We randomized 500 adults with major depressive disorder to sertraline or placebo for 8 weeks. Clinical response (>=50% HAM-D reduction) favoured sertraline (odds ratio 1.85, 95% CI 1.30 to 2.63). Nausea was more common with sertraline. Sertraline is efficacious for major depression."),
    dict(id="P021", topic="ssri_depression", title="Fluoxetine for depression: a multicenter randomized trial",
         abstract="In 700 depressed outpatients, fluoxetine 20 mg/day was compared with placebo. Response rates at 8 weeks favoured fluoxetine (odds ratio 1.60, 95% CI 1.15 to 2.23). Insomnia and headache were the most frequent side effects. Fluoxetine showed significant benefit."),
    dict(id="P022", topic="ssri_depression", title="Escitalopram in moderate depression: randomized controlled trial",
         abstract="A total of 420 patients with moderate major depressive disorder received escitalopram or placebo. The odds of response were higher with escitalopram (odds ratio 2.05, 95% CI 1.40 to 3.00). Discontinuation rates were similar between arms. Escitalopram was effective and well tolerated."),
    dict(id="P023", topic="ssri_depression", title="Paroxetine versus placebo in elderly depressed patients",
         abstract="We randomly assigned 300 older adults with major depression to paroxetine or placebo. Clinical response favoured paroxetine (odds ratio 1.45, 95% CI 0.98 to 2.14), a modest, non-significant effect. Sedation was reported more often with paroxetine. Benefit in the elderly appears limited."),
    # --- distractors: off-topic or non-numeric ---
    dict(id="D001", topic=None, title="Aspirin mechanism of action: a narrative review",
         abstract="This narrative review describes the irreversible inhibition of cyclooxygenase-1 by aspirin and downstream effects on thromboxane synthesis. No randomized outcome data are presented. The pharmacology of aspirin remains central to cardiovascular medicine."),
    dict(id="D002", topic=None, title="Exercise training in heart failure",
         abstract="In 250 patients with heart failure, supervised exercise training improved six-minute walk distance versus usual care. Quality-of-life scores also improved. Exercise training is a useful adjunct in heart failure management."),
    dict(id="D003", topic=None, title="Statin-associated muscle symptoms: a pharmacovigilance database study",
         abstract="Analysis of a national pharmacovigilance database characterized reports of muscle symptoms in statin users. Reporting rates varied by statin type. Causality assessment is limited by the observational design."),
    dict(id="D004", topic=None, title="Psychotherapy versus pill placebo in depression",
         abstract="This meta-analysis compared cognitive behavioural therapy against pill placebo control conditions in depression trials. Effect estimates were heterogeneous across studies. Blinding remains a methodological challenge."),
]

TOPIC_KEYWORDS = {
    "aspirin_mi": ["aspirin", "infarction", "myocardial"],
    "statin_ldl": ["statin", "ldl", "cholesterol", "atorvastatin", "rosuvastatin",
                   "simvastatin", "ezetimibe"],
    "ssri_depression": ["ssri", "depression", "depressive", "antidepressant",
                        "sertraline", "fluoxetine", "escitalopram", "paroxetine"],
}


def corpus_by_id():
    return {p["id"]: p for p in FIXTURE_CORPUS}


def gold_relevant(topic: str):
    """Gold set of paper ids relevant to a topic (excludes distractors)."""
    return sorted(p["id"] for p in FIXTURE_CORPUS if p["topic"] == topic)
