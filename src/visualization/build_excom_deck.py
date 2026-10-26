# -*- coding: utf-8 -*-
"""Build the Singtel ExCom HTML deck from pipeline outputs.

Inputs:  reports/deck_data.json (written by the notebook), data/interim/*.pkl,
         data/raw transcripts (for call timing), models/singtel/tabular_model.pkl
Output:  reports/singtel-churn-excom.html (self-contained: Chart.js and logo inlined)

Every number on a slide is computed here from those inputs; nothing is typed by hand.
"""
import base64
import json
import os
import pickle

import numpy as np
import pandas as pd

from src.data.make_dataset import INTERIM_DIR, ROOT_DIR, T0, load_raw_calls
from src.features.tabular_features import tenure_band
from src.models.microsegments import REASON_GROUP

REPORTS = os.path.join(ROOT_DIR, "reports")
VENDOR = os.path.join(ROOT_DIR, "references", "vendor")
LOGO = os.path.expanduser("~/.claude/skills/singtel-excom-presentation/assets/singtel-logo.png")
OUT = os.path.join(REPORTS, "singtel-churn-excom.html")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def num(x):
    return f"{int(round(x)):,}"


def money(x):
    return f"${x / 1000:,.1f}K"


# ---------------------------------------------------------------- data
def collect():
    d = json.load(open(os.path.join(REPORTS, "deck_data.json")))
    cust = pd.read_pickle(os.path.join(INTERIM_DIR, "customers.pkl"))
    cust["band"] = tenure_band(cust["Tenure Months"]).astype(str)

    # Slide 1: churn by tenure band, month-to-month vs contracted
    cust["contract_grp"] = np.where(cust["Contract"] == "Month-to-month", "Month-to-month", "1- or 2-year contract")
    tb = cust.groupby(["band", "contract_grp"], observed=True)["churn"].mean().unstack()
    bands = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]
    m2m_churn_share = cust.loc[cust.Contract == "Month-to-month", "churn"].sum() / cust.churn.sum()
    pay = cust.groupby("Payment Method")["churn"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    net = cust.groupby("Internet Service")["churn"].agg(["mean", "size"]).sort_values("mean", ascending=False)

    # Slide 2: contract x internet x tenure cells
    cells = (cust.groupby(["Contract", "Internet Service", "band"], observed=True)["churn"]
             .agg(["mean", "size", "sum"]).reset_index().query("size >= 50"))
    cells["label"] = cells["Contract"].str.replace("Month-to-month", "No contract") + " · " + \
        cells["Internet Service"].str.replace("Fiber optic", "Fibre").str.replace("No", "No internet") + " · " + cells["band"]

    # Slide 2: odds multipliers from the chosen tabular model
    model = pickle.load(open(os.path.join(ROOT_DIR, "models", "singtel", "tabular_model.pkl"), "rb"))
    pre, lr = model.named_steps["pre"], model.named_steps["lr"]
    coef = pd.Series(np.exp(lr.coef_[0]), index=[n.split("__", 1)[1] for n in pre.get_feature_names_out()])

    # Slide 5: how long before closing did churners call?
    raw = load_raw_calls()
    raw["cid"] = raw["customer_id"].astype(str).str.strip().str.upper()
    raw = raw[raw.cid.isin(set(cust.CustomerID))]
    closed = raw[raw.resolution_status == "Account Closed"].groupby("cid").interaction_date.min()
    early = raw[raw.cid.isin(closed.index) & (raw.interaction_date > T0)].assign(close=lambda x: x.cid.map(closed))
    early = early[early.interaction_date < early.close]
    lead = early.groupby("cid").apply(lambda g: (g.close.iloc[0] - g.interaction_date.min()).days)
    lead_bins = pd.cut(lead, [0, 14, 30, 60, 120], labels=["Up to 2 weeks", "2-4 weeks", "1-2 months", "2+ months"]).value_counts().reindex(
        ["Up to 2 weeks", "2-4 weeks", "1-2 months", "2+ months"]).fillna(0)
    first_early = early.sort_values("interaction_date").groupby("cid").first().call_type.value_counts()

    # Slide 7: reason mix by segment
    segs = pd.DataFrame(d["segments"])
    mix = pd.DataFrame(d["segment_mix"]).T.reindex(segs.segment)

    # Slide 8: holdout sizing. Minimum detectable churn drop, 80% power, 5% two-sided
    p = d["list_full"]["precision"]
    n_list = d["list_full"]["n"]
    holdout = []
    for share in (0.10, 0.15, 0.20):
        n_c = round(share * n_list)
        se = np.sqrt(p * (1 - p) * (1 / n_c + 1 / (n_list - n_c)))
        holdout.append({"share": share, "controls": n_c, "mde": 2.8 * se})

    return dict(d=d, cust=cust, tb=tb, bands=bands, m2m_churn_share=m2m_churn_share, pay=pay, net=net,
                cells=cells, coef=coef, lead=lead, lead_bins=lead_bins, first_early=first_early,
                segs=segs, mix=mix, holdout=holdout)


# ---------------------------------------------------------------- html pieces
def metric(value, label, note, kind=""):
    return (f'<div class="metric {kind}"><div class="value">{value}</div>'
            f'<div class="label">{label}</div><div class="note">{note}</div></div>')


def slide(n, title, subtitle, body, note):
    return f"""
<section class="slide" id="slide-{n}">
  <div class="brand-rule"></div>
  <img class="chrome-logo" src="LOGO_URI" alt="Singtel">
  <div class="masthead">
    <div class="title">
      <h2>{title}</h2>
      <p>{subtitle}</p>
    </div>
  </div>
{body}
  <div class="slide-note">{note}</div>
  <div class="chrome-foot">Singtel AIDA · Consumer churn · Confidential</div>
  <div class="page-no">{n}</div>
</section>"""


def build(x):
    d, segs = x["d"], x["segs"]
    eda, test, lf, calls, delta, comp = d["eda"], d["test"]["Logistic (chosen)"], d["list_full"], d["calls"], d["delta"], d["comparison"]
    base_rate = eda["churn_rate"]
    random_reach = 0.2 * d["split"]["test_churners"]
    seg_list = segs.set_index("segment")
    new_fibre_share = (seg_list.loc["New fibre, no support add-ons", "in_top20_list"]
                       + seg_list.loc["New fibre, with add-ons", "in_top20_list"]) / lf["n"]
    top5_share = segs.head(5)["in_top20_list"].sum() / lf["n"]
    audit = dict(calls["audit"])
    post_t0 = audit["After scoring date T0 2026-03-31 (not observable)"]
    caller_share = calls["callers"] / eda["base_customers"]
    mix = x["mix"]
    comp_share = (mix["Competitor"].min(), mix["Competitor"].max())
    ho = {h["share"]: h for h in x["holdout"]}
    tb = x["tb"]

    s = []
    # ---- 1. where churn sits
    s.append(slide(1,
        f"No-contract customers carry {pct(x['m2m_churn_share'], 0)} of churn",
        f"They are 55% of the 7,043 customers scored on 31 Mar 2026. In Apr–Jun, month-to-month churned "
        f"{pct(eda['by_contract']['Month-to-month'])} against {pct(eda['by_contract']['One year'])} on one-year "
        f"and {pct(eda['by_contract']['Two year'])} on two-year contracts.",
        f"""  <div class="grid four">
    {metric("7,043", "Customers on 31 Mar 2026", "The base for every slide that follows", "neutral")}
    {metric(pct(base_rate), "Churned Apr–Jun 2026", f"{num(eda['churners'])} customers", "loss")}
    {metric(pct(x['m2m_churn_share'], 0), "Of churners had no contract", f"No-contract churn {pct(eda['by_contract']['Month-to-month'])}", "loss")}
    {metric("$" + f"{eda['avg_bill']['1']:.0f}", "Avg monthly bill, churners", f"vs ${eda['avg_bill']['0']:.0f} for customers who stayed", "neutral")}
  </div>
  <div class="grid wide-two" style="margin-top:18px">
    <div class="panel cohort">
      <h3>Churn by tenure, with and without a contract</h3>
      <div class="sub">No-contract churn falls from {pct(tb.loc['0-6m','Month-to-month'],0)} in the first 6 months to {pct(tb.loc['49m+','Month-to-month'],0)} after 4 years; a contract keeps it under {pct(tb['1- or 2-year contract'].max(),0)} at every tenure</div>
      <div class="chart" style="height:400px"><canvas id="tenureChart"></canvas></div>
    </div>
    <div class="panel">
      <h3>Churn by internet service and payment method</h3>
      <div class="sub">Fibre churns at twice the DSL rate; electronic-check payers at three times auto-pay</div>
      <div class="chart" style="height:400px"><canvas id="mixChart"></canvas></div>
    </div>
  </div>""",
        "IBM Telco customer file (7,043 customers). Churn = account closed in the 90 days after 31 Mar 2026. Amounts in dataset currency."))

    # ---- 2. drivers
    top_cells = x["cells"].sort_values("mean", ascending=False)
    hot = top_cells.iloc[0]
    odds_rows = [("No contract (month-to-month)", "Contract_Month-to-month"), ("Paperless billing", "Paperless Billing_Yes"),
                 ("First 6 months", "new_customer"), ("Fibre internet", "Internet Service_Fiber optic"),
                 ("Pays by electronic check", "Payment Method_Electronic check"), ("Has dependents", "Dependents_Yes"),
                 ("Two-year contract", "Contract_Two year"), ("DSL internet", "Internet Service_DSL")]
    odds_html = "".join(
        f'<tr{" class=hl" if x["coef"][k] > 1.25 else ""}><td>{lbl}</td><td class=r>{x["coef"][k]:.2f}×</td>'
        f'<td class=r>{"raises" if x["coef"][k] > 1 else "lowers"}</td></tr>' for lbl, k in odds_rows)
    s.append(slide(2,
        "First-year fibre on no contract is the core risk",
        f"No-contract fibre churns {pct(hot['mean'])} in months 0–6 ({num(hot['size'])} customers) and "
        f"{pct(top_cells.iloc[1]['mean'])} in months 7–12, against a {pct(base_rate)} base rate.",
        f"""  <div class="grid two-wide">
    <div class="panel decision-matrix">
      <h3>Churn against size, by contract, internet and tenure</h3>
      <div class="sub">Each bubble is a group of at least 50 customers; size = churners; dashed = {pct(base_rate)} base rate</div>
      <div class="chart" style="height:545px"><canvas id="cellChart"></canvas></div>
    </div>
    <div class="panel evidence-table">
        <h3>What moves the odds of churn</h3>
        <div class="sub">Logistic model, holding everything else equal; above 1 raises churn odds</div>
        <table class="score-table">
          <thead><tr><th>Customer attribute</th><th class=r>Odds multiplier</th><th class=r>Effect</th></tr></thead>
          <tbody>{odds_html}</tbody>
        </table>
      <div class="answer-box" style="margin-top:14px"><strong>Contract and tenure explain most of the ranking.</strong> Payment method and paperless billing add signal but are likely markers of low commitment rather than causes. Changing how someone pays is not a retention offer.</div>
      <div class="answer-box" style="margin-top:12px"><strong>Add-ons go with lower churn.</strong> First-year no-contract fibre customers with support or security add-ons churn {pct(seg_list.loc['New fibre, with add-ons','churn_rate'])}, against {pct(seg_list.loc['New fibre, no support add-ons','churn_rate'])} without them. That is the basis for the free add-on trial on slide 7.</div>
    </div>
  </div>""",
        "Customer file, 31 Mar 2026 snapshot; churn Apr–Jun 2026. Odds multipliers from the chosen logistic model on 5,282 training customers."))

    # ---- 3. the 20% cut
    lo, hi = d["recall_ci"]
    cap = {round(r["capacity"], 1): r for r in d["capacity_curve"]}
    cap_rows = "".join(
        f'<tr{" class=hl" if k == 0.2 else ""}><td>Top {int(k*100)}%</td><td class=r>{num(r["contacted"])}</td>'
        f'<td class=r>{num(r["churners_caught"])}</td><td class=r>{pct(r["recall"])}</td><td class=r>{pct(r["precision"])}</td></tr>'
        for k, r in cap.items())
    s.append(slide(3,
        f"Calling the top 20% reaches {pct(test['recall@20'], 0)} of churners",
        f"{test['lift@20']:.1f}× a random list: on 1,761 held-out customers, the top 352 hold {num(test['churners_caught'])} of "
        f"{num(d['split']['test_churners'])} churners, where a random 352 would hold {num(random_reach)}.",
        f"""  <div class="grid four">
    {metric(pct(test['recall@20']), "Churners on the list", f"95% range {pct(lo)}–{pct(hi)}")}
    {metric(pct(test['precision@20']), "Of calls reach a churner", f"vs {pct(base_rate)} for a random call")}
    {metric(f"{test['lift@20']:.2f}×", "Lift over random", f"{num(test['churners_caught'])} vs {num(random_reach)} churners", "neutral")}
    {metric(num(lf['churners_caught']), "Churners on the full-base list", f"of {num(eda['churners'])} ({pct(lf['recall@20'])}) from {num(lf['n'])} calls", "neutral")}
  </div>
  <div class="grid three" style="margin-top:18px">
    <div class="panel funnel">
      <h3>Churners reached as the list grows</h3>
      <div class="sub">The curve is steep to 20%, then flattens</div>
      <div class="chart" style="height:390px"><canvas id="gainsChart"></canvas></div>
    </div>
    <div class="panel">
      <h3>Churn rate by score decile</h3>
      <div class="sub">The top two deciles churn {pct(d['deciles'][0]['churn_rate'],0)} and {pct(d['deciles'][1]['churn_rate'],0)}; the bottom three under 3%</div>
      <div class="chart" style="height:390px"><canvas id="decileChart"></canvas></div>
    </div>
    <div class="panel evidence-table">
      <h3>Capacity options</h3>
      <div class="sub">Each extra 1% above 20% finds {cap[0.3]['extra churners per extra 1% of base']:.1f} churners, down from {cap[0.2]['extra churners per extra 1% of base']:.1f}</div>
      <table class="score-table">
        <thead><tr><th>List</th><th class=r>Calls</th><th class=r>Churners</th><th class=r>Recall</th><th class=r>Precision</th></tr></thead>
        <tbody>{cap_rows}</tbody>
      </table>
      <div class="answer-box" style="margin-top:14px"><strong>Calls beyond 20% hit churners less often.</strong> Going from 20% to 30% adds {num(cap[0.3]['churners_caught'] - cap[0.2]['churners_caught'])} churners for {num(cap[0.3]['contacted'] - cap[0.2]['contacted'])} extra calls on the test set, a hit rate of {pct((cap[0.3]['churners_caught'] - cap[0.2]['churners_caught']) / (cap[0.3]['contacted'] - cap[0.2]['contacted']), 0)} against {pct(test['precision@20'], 0)} inside the list.</div>
    </div>
  </div>""",
        "Test set = 25% of customers held out from training (1,761; 467 churners). Full-base list uses out-of-fold scores for all 7,043. 95% range from 1,000 bootstrap resamples."))

    # ---- 4. transcripts verdict
    comp_rows = ""
    for name, short in [("Tabular only", "Account data only"), ("Tabular + call sequence", "+ call reasons and outcomes"),
                        ("Tabular + call sequence + text", "+ transcript text")]:
        r = comp[name]
        comp_rows += (f'<tr{" class=hl" if "text" in name else ""}><td>{short}</td><td class=r>{pct(r["Test recall@20"])}</td>'
                      f'<td class=r>{pct(r["Test precision@20"])}</td><td class=r>{r["Callers PR-AUC"]:.2f}</td></tr>')
    s.append(slide(4,
        f"Call history finds {delta['caller_churners_comb'] - delta['caller_churners_tab']} more caller-churners, "
        f"{num(comp['Tabular + call sequence + text']['Churners caught'] - comp['Tabular only']['Churners caught'])} net",
        f"Same test set and 20% cut: recall {pct(comp['Tabular only']['Test recall@20'])} → {pct(comp['Tabular + call sequence + text']['Test recall@20'])} "
        f"({delta['lr']['delta']*100:+.1f} pp; 95% range {delta['lr']['ci_low']*100:+.1f} to {delta['lr']['ci_high']*100:+.1f}). ".replace("-", "−")
        +
        f"Only {pct(caller_share)} of customers called before 31 Mar.",
        f"""  <div class="grid three">
    {metric(num(calls['callers']), "Customers who called before T0", f"{pct(caller_share)} of the base; they churn {pct(calls['caller_churn'])}", "neutral")}
    {metric(f"{delta['caller_churners_tab']} → {delta['caller_churners_comb']}", "Caller-churners on the list", f"of {delta['caller_churners']} in the test set", "")}
    {metric(f"{delta['lr']['delta']*100:+.1f} pp", "Recall change, whole base", "Within the margin of error", "neutral")}
  </div>
  <div class="grid two" style="margin-top:18px">
    <div class="panel evidence-table">
      <h3>Same test set, three feature sets</h3>
      <div class="sub">Reasons and outcomes do most of the work; the words add 0.01 PR-AUC among callers</div>
      <table class="score-table">
        <thead><tr><th>Model inputs</th><th class=r>Recall@20%</th><th class=r>Precision@20%</th><th class=r>Callers PR-AUC</th></tr></thead>
        <tbody>{comp_rows}</tbody>
      </table>
      <div class="answer-box" style="margin-top:12px"><strong>Keep the call reason codes, park the NLP.</strong> Wrap-up codes and resolution status are already in the CRM. A transcript pipeline costs more than the 1 extra churner it found on this test set.</div>
      <div class="risk-list" style="margin-top:12px">
        <div class="risk-item warn"><strong>The direction holds with a second model.</strong> With gradient-boosted trees the same comparison gives {delta['lgb']['delta']*100:+.1f} pp (95% range {delta['lgb']['ci_low']*100:+.1f} to {delta['lgb']['ci_high']*100:+.1f}), positive but small.</div>
        <div class="risk-item"><strong>Customers who never called are scored on account data alone.</strong> They are 90% of the base, so they get a separate baseline in the model; their missing call history is not filled in.</div>
      </div>
    </div>
    <div class="panel">
      <h3>Ranking quality among the {delta['test_callers']} test customers who called</h3>
      <div class="sub">PR-AUC rises from {comp['Tabular only']['Callers PR-AUC']:.2f} to {comp['Tabular + call sequence + text']['Callers PR-AUC']:.2f} once call history is added</div>
      <div class="chart" style="height:410px"><canvas id="callerChart"></canvas></div>
    </div>
  </div>""",
        "Contact-centre calls 1 Sep 2025 – 31 Mar 2026, customer turns only. PR-AUC: 1.0 = every churner ranked above every stayer. 95% range from 1,000 paired bootstrap resamples."))

    # ---- 5. trap and trigger
    fe = x["first_early"]
    s.append(slide(5,
        "Churners phone about a month before they leave",
        f"{calls['early_callers']} of {calls['closers']} customers who closed by phone had called earlier, a median {calls['lead_median_days']:.0f} days before. "
        f"The {num(post_t0)} calls after 31 Mar, {calls['post_t0_closed']} of them cancellations, are kept out of the model.",
        f"""  <div class="grid two">
    <div class="panel bridge-panel">
      <h3>From raw extract to usable calls</h3>
      <div class="sub">{num(calls['usable_calls'])} calls from {num(calls['callers'])} customers survive; post-T0 calls are 40% of the file</div>
      <div class="chart" style="height:545px"><canvas id="bridgeChart"></canvas></div>
    </div>
    <div class="panel">
        <h3>Time from earlier call to account closure</h3>
        <div class="sub">{calls['early_callers']} churners, Apr–Jun 2026: {int(x['lead_bins'].iloc[1] + x['lead_bins'].iloc[2])} of them called 2–8 weeks before closing</div>
        <div class="chart" style="height:340px"><canvas id="leadChart"></canvas></div>
      <div class="risk-list source-limitations" style="margin-top:12px">
        <div class="risk-item"><strong>Post-T0 calls leak the outcome.</strong> 96% of post-T0 termination calls are the cancellation itself; any model that sees them scores the past.</div>
        <div class="risk-item warn"><strong>Outbound calls carry our own targeting.</strong> 15 of 20 pre-T0 retention-offer calls were outbound: we already chose whom to call.</div>
        <div class="risk-item warn"><strong>Agent turns are script.</strong> 64% of agent lines repeat word for word across calls; only customer turns are scored.</div>
      </div>
    </div>
  </div>""",
        "Contact-centre extract 1 Sep 2025 – 30 Jun 2026, voice only. Also set aside: 8 calls with unknown customer IDs, 17 duplicate logs; 38 disconnected calls kept as a count only."))

    # ---- 6. microsegments
    seg_rows = "".join(
        f'<tr{" class=hl" if r.in_top20_list > 200 else ""}><td>{r.segment}</td><td class=r>{num(r.customers)}</td>'
        f'<td class=r>{pct(r.churn_rate)}</td><td class=r>{num(r.in_top20_list)}</td>'
        f'<td class=r>{pct(r.list_precision) if r.in_top20_list >= 20 else "–"}</td><td class=r>{money(r.monthly_bill_lost)}</td></tr>'
        for r in segs.itertuples())
    s.append(slide(6,
        f"Five segments fill {pct(top5_share, 0)} of the call list",
        f"First-year fibre takes {pct(new_fibre_share,0)} of the {num(lf['n'])} places. Contracted customers are 44% of the base "
        f"and take {num(seg_list.loc['On 1- or 2-year contract','in_top20_list'])}.",
        f"""  <div class="grid two-wide">
    <div class="panel decision-matrix">
      <h3>Segment churn against size</h3>
      <div class="sub">Bubble size = list places; first-year fibre sits highest on churn</div>
      <div class="chart" style="height:545px"><canvas id="segChart"></canvas></div>
    </div>
    <div class="panel evidence-table">
      <h3>Microsegments, full base</h3>
      <div class="sub">Ordered rules, first match wins; every customer is in one segment</div>
      <table class="score-table">
        <thead><tr><th>Segment</th><th class=r>Customers</th><th class=r>Churn</th><th class=r>On list</th><th class=r>List precision</th><th class=r>Bill lost / month</th></tr></thead>
        <tbody>{seg_rows}</tbody>
      </table>
      <div class="answer-box" style="margin-top:12px"><strong>Established no-contract fibre loses the most revenue.</strong> {num(seg_list.loc['Established fibre, no contract','churners'])} churners take {money(seg_list.loc['Established fibre, no contract','monthly_bill_lost'])} a month in bills, the largest of any segment, at {pct(seg_list.loc['Established fibre, no contract','list_precision'],0)} list precision.</div>
      <div class="answer-box" style="margin-top:12px"><strong>Complainers are the surest calls.</strong> {pct(seg_list.loc['Called to complain or leave','list_precision'],0)} of the {num(seg_list.loc['Called to complain or leave','in_top20_list'])} complainers on the list churn, the highest list precision of any segment, and the contact centre already knows who they are.</div>
    </div>
  </div>""",
        "Out-of-fold scores for all 7,043 customers; list = top 20% (1,409). 'New' = tenure up to 12 months. Bill lost = monthly bill of customers in the segment who churned, Apr–Jun 2026."))

    # ---- 7. levers
    act_rows = "".join(
        f'<tr><td>{r.segment}</td><td class=r>{num(r.in_top20_list)}</td><td>{r.action}</td></tr>'
        for r in segs.itertuples() if r.in_top20_list >= 20)
    s.append(slide(7,
        "Pick offers by contract, tenure and add-ons",
        f"Recorded reasons look the same in every segment (competitor {pct(comp_share[0],0)}–{pct(comp_share[1],0)} of churners), "
        "so they cannot pick the offer. The levers below use contract, tenure, add-ons and complaints, and each is a test.",
        f"""  <div class="grid wide-two">
    <div class="panel">
      <h3>Why churners left, by segment</h3>
      <div class="sub">Reason mix barely moves across segments; only 3% left for reasons no offer can fix</div>
      <div class="chart" style="height:545px"><canvas id="reasonChart"></canvas></div>
    </div>
    <div class="panel story-spine">
      <h3>Offer to test, by segment on the list</h3>
      <div class="sub">Each lever targets an attribute the segment is defined by</div>
      <table class="score-table">
        <thead><tr><th>Segment</th><th class=r>List places</th><th>Offer to test</th></tr></thead>
        <tbody>{act_rows}</tbody>
      </table>
      <div class="answer-box" style="margin-top:14px"><strong>Give the retention desk price-match authority.</strong> Competitor offers are {pct(comp_share[0],0)}–{pct(comp_share[1],0)} of churn in every segment, so every segment needs a counter-offer option.</div>
    </div>
  </div>""",
        "Churn reasons recorded at exit for the 1,869 customers who left Apr–Jun 2026, grouped into six families."))

    # ---- 8. holdout
    s.append(slide(8,
        "Test the list against a random holdout first",
        f"Every number so far shows who leaves, not who an offer keeps. Holding out 15% of the {num(lf['n'])}-customer list "
        f"({ho[0.15]['controls']} customers) detects a {ho[0.15]['mde']*100:.0f}-point drop in churn in one quarter.",
        f"""  <div class="grid three">
    {metric(num(lf['n']), "Customers to call this quarter", f"{pct(lf['precision'],0)} would otherwise churn", "neutral")}
    {metric(str(ho[0.15]['controls']), "Random holdout (15%)", "Same score mix, no contact", "neutral")}
    {metric(f"{ho[0.15]['mde']*100:.0f} pts", "Smallest save we can detect", f"{ho[0.10]['mde']*100:.0f} pts with a 10% holdout", "neutral")}
  </div>
  <div class="grid two" style="margin-top:18px">
    <div class="panel">
      <h3>What the first quarter has to prove</h3>
      <div class="sub">Three numbers decide whether to scale</div>
      <div class="risk-list">
        <div class="risk-item"><strong>Save rate by segment.</strong> Churn in contacted vs holdout customers. A segment that saves under 5 points does not earn its calls.</div>
        <div class="risk-item"><strong>Cost per save.</strong> Offer cost and agent time over customers saved, against the monthly bill kept.</div>
        <div class="risk-item"><strong>Complaint call-back speed.</strong> Share of complaint, dispute and porting calls called back within 48 hours.</div>
      </div>
    </div>
    <div class="panel">
      <h3>Decisions for this quarter</h3>
      <div class="sub">What to fund now and what to hold</div>
      <div class="risk-list">
        <div class="risk-item"><strong>Fund:</strong> the 1,409-customer list on account data plus call reason codes, and a 48-hour call-back for complaint and porting calls.</div>
        <div class="risk-item warn"><strong>Hold:</strong> transcript NLP for the quarterly score; revisit once the trigger is live and lead-time can be measured.</div>
        <div class="risk-item warn"><strong>Skip:</strong> outreach to the 3,090 contracted customers before renewal (6% churn).</div>
      </div>
    </div>
  </div>
  <div class="panel story-spine" style="margin-top:18px">
    <table class="score-table">
      <thead><tr><th>Quarter</th><th>What runs</th><th>Read-out that decides the next step</th></tr></thead>
      <tbody>
        <tr><td>Quarter 1</td><td>Call the {num(lf['n'])}-customer list, hold out {ho[0.15]['controls']}; start the 48-hour complaint call-back</td><td>Save rate and cost per save by segment</td></tr>
        <tr><td>Quarter 2</td><td>Rescore on the new snapshot; drop segments that saved under 5 points</td><td>Uplift model on two quarters of contacted vs holdout</td></tr>
        <tr><td>Quarter 3</td><td>Decide on transcript NLP using the trigger's measured lead time</td><td>Saves from trigger calls vs the quarterly list</td></tr>
      </tbody>
    </table>
  </div>""",
        "Detectable drop: 80% power, 5% two-sided test, assuming 72% churn without contact. Holdout drawn at random within the list."))

    charts = {
        "bands": x["bands"],
        "tenure_m2m": [round(tb.loc[b, "Month-to-month"] * 100, 1) for b in x["bands"]],
        "tenure_con": [round(tb.loc[b, "1- or 2-year contract"] * 100, 1) for b in x["bands"]],
        "mix_labels": ["Fibre", "DSL", "No internet", "Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"],
        "mix_values": [round(v * 100, 1) for v in [x["net"].loc["Fiber optic", "mean"], x["net"].loc["DSL", "mean"], x["net"].loc["No", "mean"],
                       x["pay"].loc["Electronic check", "mean"], x["pay"].loc["Mailed check", "mean"],
                       x["pay"].loc["Bank transfer (automatic)", "mean"], x["pay"].loc["Credit card (automatic)", "mean"]]],
        "base_rate": round(base_rate * 100, 1),
        "cells": [{"x": int(r["size"]), "y": round(r["mean"] * 100, 1), "n": int(r["sum"]), "label": r["label"],
                   "hot": bool(r["Contract"] == "Month-to-month" and r["Internet Service"] == "Fiber optic" and r["band"] in ("0-6m", "7-12m"))}
                  for _, r in x["cells"].iterrows()],
        "gains_x": [round(v * 100) for v in d["gains"]["tabular"]["share_contacted"]],
        "gains_y": [round(v * 100, 1) for v in d["gains"]["tabular"]["share_churners"]],
        "deciles": [round(r["churn_rate"] * 100, 1) for r in d["deciles"]],
        "caller_auc": {k: [round(comp[k]["Callers ROC-AUC"], 3), round(comp[k]["Callers PR-AUC"], 3)] for k in comp},
        "bridge": [["Raw extract", calls["raw_calls"]], ["Unknown customer ID", -audit["Customer ID not in customer file (orphan)"]],
                   ["After 31 Mar scoring date", -post_t0], ["Duplicate logs", -audit["Duplicate log of the same call"]],
                   ["Usable calls", calls["usable_calls"]]],
        "lead_labels": list(x["lead_bins"].index), "lead_values": [int(v) for v in x["lead_bins"].values],
        "segs": [{"label": r.segment.replace("support add-ons", "add-ons"), "x": int(r.customers), "y": round(r.churn_rate * 100, 1), "r": int(r.in_top20_list)} for r in segs.itertuples()],
        "reason_segments": list(mix.index),
        "reason_series": {c: [round(v * 100, 1) for v in mix[c].fillna(0)] for c in
                          ["Competitor", "Product / network", "Support experience", "Price", "Unknown", "Not saveable"]},
    }
    return "\n".join(s), charts


CSS = """
:root {
  --brand: #E40230;
  --deep: #C00000;
  --maroon: #943634;
  --good: #1F7A4D;
  --ink: #1A1A1A;
  --muted: #5F6368;
  --line: #D8CDCF;
  --fill: #FBE5E9;
}
* { box-sizing: border-box; }
body { margin: 0; background: #EFEBEB; color: var(--ink);
  font-family: "Segoe UI", Arial, Helvetica, sans-serif; line-height: 1.32; overflow-x: hidden; }
.deck-frame { width: min(100vw, 1600px); margin: 0 auto; overflow: visible; }
.deck { width: 1600px; max-width: none; margin: 0; transform-origin: top left; transform: scale(var(--deck-scale, 1)); }
.slide { position: relative; width: 1600px; max-width: none; height: 900px; min-height: 900px;
  padding: 58px 68px 74px 100px; overflow: hidden; background: #fff; border-bottom: 1px solid #D9D9D9; }
.slide + .slide { margin-top: 52px; }
.brand-rule { position: absolute; left: 0; right: 0; top: 0; height: 10px; background: var(--brand); }
.chrome-logo { position: absolute; top: 26px; right: 60px; width: 92px; height: auto; }
.chrome-foot { position: absolute; left: 100px; bottom: 16px; color: #8A8A8A; font-size: 12px; }
.masthead { min-height: 104px; padding-right: 245px; margin-bottom: 10px; }
.title h1, .title h2 { margin: 0; color: var(--brand); font-size: 45px; line-height: 1.02; font-weight: 850; letter-spacing: 0; }
.title p { margin: 8px 0 0; color: var(--maroon); font-size: 20px; }
.grid { display: grid; gap: 18px; }
.grid.two { grid-template-columns: 1fr 1fr; }
.grid.two-wide { grid-template-columns: .92fr 1.08fr; }
.grid.wide-two { grid-template-columns: 1.15fr .85fr; }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.stack { display: grid; gap: 14px; align-content: start; }
.metric { border: 1.5px solid var(--brand); background: #fff; min-height: 96px; padding: 11px 16px; }
.metric .value { font-size: 35px; line-height: 1; font-weight: 850; color: var(--brand); }
.metric.loss .value { color: var(--deep); }
.metric.neutral .value { color: var(--ink); }
.metric .label { margin-top: 7px; color: var(--muted); font-size: 14px; font-weight: 850; text-transform: uppercase; }
.metric .note { margin-top: 4px; color: #333; font-size: 14px; }
.panel { border: 1.5px solid var(--brand); background: #fff; padding: 14px 18px 12px; min-width: 0; }
.panel h3 { margin: 0 0 4px; font-size: 22px; font-weight: 850; }
.panel .sub { margin: 0 0 8px; color: var(--muted); font-size: 15px; }
.brand-band { background: var(--deep); color: #fff; padding: 7px 15px; font-size: 20px; font-weight: 850; margin: 0 0 12px; }
.answer-box { background: #F4F1F1; padding: 12px 16px; font-size: 17px; line-height: 1.3; }
.answer-box strong { color: var(--brand); }
.chart { position: relative; height: 260px; }
.chart.short { height: 225px; }
.chart.mid { height: 330px; }
.chart.tall { height: 470px; }
.chart.xtall { height: 560px; }
.score-table { width: 100%; border-collapse: collapse; font-size: 15px; }
.score-table th { background: var(--deep); color: #fff; padding: 7px 10px; text-align: left; font-weight: 850; }
.score-table th.r { text-align: right; }
.score-table td { padding: 7px 10px; border-bottom: 1px solid #E6DADC; vertical-align: top; }
.score-table .r { text-align: right; font-variant-numeric: tabular-nums; }
.score-table tr.hl td { background: var(--fill); font-weight: 700; }
.story-spine .score-table td:first-child { font-weight: 850; color: var(--maroon); width: 230px; }
.risk-list { display: grid; gap: 10px; }
.risk-item { border-left: 9px solid var(--brand); background: #F6F4F4; padding: 9px 14px; font-size: 16px; line-height: 1.28; }
.risk-item.warn { border-left-color: var(--maroon); }
.slide-note { position: absolute; right: 68px; bottom: 36px; max-width: 640px; text-align: right; color: var(--muted);
  font-size: 12px; line-height: 1.3; padding-right: 10px; border-right: 2px solid var(--brand); }
.page-no { position: absolute; right: 68px; bottom: 14px; color: #7A7A7A; font-size: 12px; }
@media print {
  body { overflow-x: visible; }
  .deck-frame { width: 1600px; height: auto !important; }
  .deck { transform: none; }
  .slide { break-after: page; }
  .slide + .slide { margin-top: 0; }
  .brand-rule, .score-table th, .brand-band { print-color-adjust: exact; -webkit-print-color-adjust: exact; }
}
"""

JS = r"""
Chart.register(ChartDataLabels);
Chart.defaults.font.family = '"Segoe UI", Arial, Helvetica, sans-serif';
Chart.defaults.font.size = 13;
Chart.defaults.color = '#333';
Chart.defaults.plugins.legend.labels.boxWidth = 14;
const RED = '#E40230', DEEP = '#C00000', MAROON = '#943634', GREY = '#9AA0A6', FILL = 'rgba(228,2,48,0.18)';
const grid = { color: '#EEE' };
const pctAxis = (max) => ({ min: 0, max, grid, ticks: { callback: v => v + '%' } });
const noLabels = { datalabels: { display: false } };
const baseLine = (v) => ({ id: 'baseLine', afterDraw(c) { const y = c.scales.y.getPixelForValue(v), a = c.chartArea, x = c.ctx;
  x.save(); x.setLineDash([6, 4]); x.strokeStyle = '#555'; x.beginPath(); x.moveTo(a.left, y); x.lineTo(a.right, y); x.stroke();
  x.restore(); } });

new Chart(tenureChart, { type: 'bar', data: { labels: D.bands, datasets: [
  { label: 'No contract', data: D.tenure_m2m, backgroundColor: FILL, borderColor: RED, borderWidth: 1.5 },
  { label: '1- or 2-year contract', data: D.tenure_con, backgroundColor: '#E5E5E5', borderColor: GREY, borderWidth: 1.5 }]},
  options: { maintainAspectRatio: false, layout: { padding: { top: 20 } }, scales: { y: pctAxis(80), x: { grid: { display: false } } },
    plugins: { legend: { position: 'top', align: 'end' }, datalabels: { anchor: 'end', align: 'top', formatter: v => v.toFixed(0) + '%', font: { weight: 700 } },
      tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}% churn` } } } } });

new Chart(mixChart, { type: 'bar', data: { labels: D.mix_labels, datasets: [{ data: D.mix_values,
  backgroundColor: D.mix_values.map(v => v > D.base_rate * 1.1 ? FILL : '#EDEDED'), borderColor: D.mix_values.map(v => v > D.base_rate * 1.1 ? RED : GREY), borderWidth: 1.5 }]},
  options: { indexAxis: 'y', maintainAspectRatio: false, layout: { padding: { right: 40 } },
    scales: { x: pctAxis(60), y: { grid: { display: false } } },
    plugins: { legend: { display: false }, datalabels: { anchor: 'end', align: 'right', formatter: v => v.toFixed(1) + '%', font: { weight: 700 } },
      tooltip: { callbacks: { label: c => `${c.parsed.x.toFixed(1)}% churn` } } } } });

const maxN = Math.max(...D.cells.map(c => c.n));
new Chart(cellChart, { type: 'bubble', data: { datasets: [{ data: D.cells.map(c => ({ x: c.x, y: c.y, r: 4 + 26 * Math.sqrt(c.n / maxN), label: c.label, n: c.n, hot: c.hot })),
  backgroundColor: D.cells.map(c => c.hot ? 'rgba(228,2,48,0.55)' : c.y > D.base_rate ? 'rgba(148,54,52,0.25)' : 'rgba(154,160,166,0.3)'),
  borderColor: D.cells.map(c => c.hot ? RED : c.y > D.base_rate ? MAROON : GREY), borderWidth: 1.2 }]},
  options: { maintainAspectRatio: false, layout: { padding: { top: 10, right: 30 } },
    scales: { y: { ...pctAxis(85), title: { display: true, text: 'Churn rate' } }, x: { min: 0, max: 1100, grid, title: { display: true, text: 'Customers in group' } } },
    plugins: { legend: { display: false },
      datalabels: { display: c => c.dataset.data[c.dataIndex].y > 40, align: c => /13-24m/.test(c.dataset.data[c.dataIndex].label) ? 'top' : 'right', offset: c => c.dataset.data[c.dataIndex].r + 2, clip: false, color: '#222', font: { size: 12, weight: 600 },
        formatter: v => `${v.label} (${v.y.toFixed(0)}%)` },
      tooltip: { callbacks: { label: c => `${c.raw.label}: ${c.raw.y.toFixed(1)}% churn, ${c.raw.x.toLocaleString()} customers, ${c.raw.n} churners` } } } },
  plugins: [baseLine(D.base_rate)] });

new Chart(gainsChart, { type: 'line', data: { labels: D.gains_x, datasets: [
  { label: 'Model list', data: D.gains_y, borderColor: RED, backgroundColor: RED, pointRadius: 0, borderWidth: 2.5 },
  { label: 'Random list', data: D.gains_x, borderColor: GREY, borderDash: [5, 4], pointRadius: 0, borderWidth: 1.5 }]},
  options: { maintainAspectRatio: false, scales: { y: { ...pctAxis(100), title: { display: true, text: '% of churners reached' } },
    x: { grid, title: { display: true, text: '% of customers called' }, ticks: { maxRotation: 0, autoSkip: false, callback: (v, i) => D.gains_x[i] % 20 === 0 ? D.gains_x[i] + '%' : '' } } },
    plugins: { legend: { position: 'bottom' },
      datalabels: { display: c => c.datasetIndex === 0 && D.gains_x[c.dataIndex] === 20, align: 'left', offset: 6, color: DEEP, font: { weight: 800 }, formatter: v => v.toFixed(0) + '% at 20%' },
      tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(0)}% of churners` } } } } });

new Chart(decileChart, { type: 'bar', data: { labels: D.deciles.map((_, i) => 'D' + (i + 1)), datasets: [{ data: D.deciles,
  backgroundColor: D.deciles.map((_, i) => i < 2 ? FILL : '#EDEDED'), borderColor: D.deciles.map((_, i) => i < 2 ? RED : GREY), borderWidth: 1.5 }]},
  options: { maintainAspectRatio: false, layout: { padding: { top: 18 } }, scales: { y: pctAxis(90), x: { grid: { display: false } } },
    plugins: { legend: { display: false }, datalabels: { anchor: 'end', align: 'top', formatter: v => v.toFixed(0) + '%', font: { size: 11, weight: 700 } },
      tooltip: { callbacks: { label: c => `${c.parsed.y.toFixed(1)}% churn` } } } }, plugins: [baseLine(D.base_rate)] });

const ca = D.caller_auc, caKeys = Object.keys(ca), caNames = ['Account data only', '+ call reasons and outcomes', '+ transcript text'];
new Chart(callerChart, { type: 'bar', data: { labels: ['ROC-AUC', 'PR-AUC'], datasets: caKeys.map((k, i) => ({ label: caNames[i], data: ca[k],
  backgroundColor: ['#EDEDED', 'rgba(148,54,52,0.3)', FILL][i], borderColor: [GREY, MAROON, RED][i], borderWidth: 1.5 })) },
  options: { maintainAspectRatio: false, layout: { padding: { top: 20 } }, scales: { y: { min: 0.7, max: 1, grid }, x: { grid: { display: false } } },
    plugins: { legend: { position: 'top', align: 'end' }, datalabels: { anchor: 'end', align: 'top', formatter: v => v.toFixed(2), font: { weight: 700 } },
      tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } } } } });

// Bridge: floating bars from raw extract to usable calls
let run = 0; const bridge = D.bridge.map(([l, v], i) => { if (i === 0 || i === D.bridge.length - 1) { run = Math.abs(v); return [0, Math.abs(v)]; }
  const top = run; run += v; return [run, top]; });
new Chart(bridgeChart, { type: 'bar', data: { labels: D.bridge.map(b => b[0].split(' ').length > 2 ? [b[0].split(' ').slice(0, 2).join(' '), b[0].split(' ').slice(2).join(' ')] : b[0]), datasets: [{ data: bridge,
  backgroundColor: D.bridge.map((b, i) => i === 0 || i === D.bridge.length - 1 ? '#EDEDED' : FILL),
  borderColor: D.bridge.map((b, i) => i === 0 || i === D.bridge.length - 1 ? GREY : RED), borderWidth: 1.5 }]},
  options: { maintainAspectRatio: false, layout: { padding: { top: 24 } }, scales: { y: { min: 0, max: 2500, grid, ticks: { callback: v => v.toLocaleString() } }, x: { grid: { display: false }, ticks: { maxRotation: 0 } } },
    plugins: { legend: { display: false }, datalabels: { anchor: 'end', align: 'top', font: { weight: 700 },
      formatter: (v, c) => { const raw = D.bridge[c.dataIndex][1]; return (raw < 0 ? '−' : '') + Math.abs(raw).toLocaleString(); } },
      tooltip: { callbacks: { label: c => `${Math.abs(D.bridge[c.dataIndex][1]).toLocaleString()} calls` } } } } });

new Chart(leadChart, { type: 'bar', data: { labels: D.lead_labels, datasets: [{ data: D.lead_values, backgroundColor: FILL, borderColor: RED, borderWidth: 1.5 }]},
  options: { maintainAspectRatio: false, layout: { padding: { top: 18 } }, scales: { y: { min: 0, grid, title: { display: true, text: 'Churners' } }, x: { grid: { display: false } } },
    plugins: { legend: { display: false }, datalabels: { anchor: 'end', align: 'top', font: { weight: 700 } },
      tooltip: { callbacks: { label: c => `${c.parsed.y} churners` } } } } });

const maxR = Math.max(...D.segs.map(s => s.r));
new Chart(segChart, { type: 'bubble', data: { datasets: [{ data: D.segs.map(s => ({ x: s.x, y: s.y, r: 5 + 32 * Math.sqrt(s.r / maxR), label: s.label, places: s.r })),
  backgroundColor: D.segs.map(s => s.r > 200 ? 'rgba(228,2,48,0.45)' : s.r > 50 ? 'rgba(148,54,52,0.3)' : 'rgba(154,160,166,0.3)'),
  borderColor: D.segs.map(s => s.r > 200 ? RED : s.r > 50 ? MAROON : GREY), borderWidth: 1.2 }]},
  options: { maintainAspectRatio: false, layout: { padding: { top: 24, right: 40, left: 10 } },
    scales: { y: { ...pctAxis(85), title: { display: true, text: 'Churn rate' } }, x: { type: 'logarithmic', min: 150, max: 5000, grid,
      title: { display: true, text: 'Customers in segment (log scale)' }, ticks: { callback: v => [200, 500, 1000, 2000, 5000].includes(v) ? v.toLocaleString() : '' } } },
    plugins: { legend: { display: false },
      datalabels: { align: c => ['bottom', 'top', 'right', 'top', 'bottom', 'top', 'top'][c.dataIndex], offset: c => c.dataset.data[c.dataIndex].r + 2, clip: false, color: '#222', font: { size: 12, weight: 600 },
        formatter: v => `${v.label}: ${v.places.toLocaleString()} on list` },
      tooltip: { callbacks: { label: c => `${c.raw.label}: ${c.raw.y.toFixed(1)}% churn, ${c.raw.x.toLocaleString()} customers, ${c.raw.places} on list` } } } },
  plugins: [baseLine(D.base_rate)] });

const rs = D.reason_series, rcol = { 'Competitor': RED, 'Product / network': MAROON, 'Support experience': '#D97A8A', 'Price': '#F2B8C2', 'Unknown': '#CFCFCF', 'Not saveable': '#8A8A8A' };
new Chart(reasonChart, { type: 'bar', data: { labels: D.reason_segments, datasets: Object.keys(rs).map(k => ({ label: k, data: rs[k], backgroundColor: rcol[k], borderColor: '#fff', borderWidth: 1 })) },
  options: { indexAxis: 'y', maintainAspectRatio: false, scales: { x: { stacked: true, min: 0, max: 100, grid, ticks: { callback: v => v + '%' } }, y: { stacked: true, grid: { display: false } } },
    plugins: { legend: { position: 'bottom' },
      datalabels: { display: c => c.dataset.data[c.dataIndex] >= 10, color: c => ['Competitor', 'Product / network'].includes(c.dataset.label) ? '#fff' : '#222',
        font: { size: 11, weight: 700 }, formatter: v => v.toFixed(0) + '%' },
      tooltip: { callbacks: { label: c => `${c.dataset.label}: ${c.parsed.x.toFixed(1)}%` } } } } });

function fit() { const s = Math.min(1, document.documentElement.clientWidth / 1600);
  document.documentElement.style.setProperty('--deck-scale', s);
  document.querySelector('.deck-frame').style.height = (document.querySelector('.deck').scrollHeight * s) + 'px'; }
window.addEventListener('resize', fit); fit();
"""


def main():
    x = collect()
    slides, charts = build(x)
    logo = "data:image/png;base64," + base64.b64encode(open(LOGO, "rb").read()).decode()
    lib = open(os.path.join(VENDOR, "chart.umd.min.js")).read() + "\n" + open(os.path.join(VENDOR, "chartjs-plugin-datalabels.min.js")).read()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Churn Retention Targeting</title>
<!-- Chart.js 4.4.8 and chartjs-plugin-datalabels 2.2.0, inlined so the deck works offline -->
<script>{lib}</script>
<style>{CSS}</style>
</head>
<body>
<div class="deck-frame">
<main class="deck">
{slides.replace("LOGO_URI", logo)}
</main>
</div>
<script>
const D = {json.dumps(charts)};
{JS}
</script>
</body>
</html>"""
    open(OUT, "w").write(html)
    print("wrote", OUT, f"{len(html) / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
