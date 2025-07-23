import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pickle
import networkx as nx
from agg_score import aggression_score

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
try:
    st.set_page_config(page_title="T20 Entry Planner", layout="wide")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────────────────────────────────────
THEME_RED = "#ff0000"
THEME_DARK = "#1c1c1c"
THEME_LIGHT = "#f5f5f7"

BASE_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
  font-family: 'Inter', sans-serif !important;
  color: #222;
}}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Grid wrapper */
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 18px;
  margin-top: 12px;
}}

/* Card */
.card {{
  background: #ffffff;
  border-radius: 16px;
  padding: 18px 20px 16px;
  box-shadow: 0 6px 22px rgba(0,0,0,.07);
  border: 1px solid #ececec;
}}
.card-header {{
  font-size: 1.05rem;
  font-weight: 600;
  color: {THEME_RED};
  margin-bottom: 10px;
  letter-spacing: 0.2px;
}}
.name {{
  font-weight: 600;
  font-size: 1rem;
  margin: 0 0 8px 0;
}}
.meta {{
  opacity: .80;
  font-size: 0.88rem;
  line-height: 1.1rem;
  margin-bottom: 4px;
}}

/* Bars */
.bar-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0 6px 0;
}}
.bar-label {{
  width: 120px;
  font-size: 0.8rem;
  color: #444;
}}
.bar-val {{
  font-size: 0.8rem;
  color: #111;
  width: 52px;
  text-align: right;
}}
.bar {{
  flex: 1;
  background: #e6e6e6;
  border-radius: 6px;
  height: 8px;
  overflow: hidden;
}}
.bar-fill {{
  height: 100%;
  border-radius: 6px;
}}
.bar-fill.avg {{ background: {THEME_RED}; }}
.bar-fill.pace {{ background: #004c97; }}   /* dark blue */
.bar-fill.spin {{ background: #2ca02c; }}   /* green */

/* Titles */
.section-title {{ font-size: 1.2rem; font-weight: 600; margin: 4px 0 12px 0; color:#111; }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# HTML builders
# ─────────────────────────────────────────────────────────────────────────────
import math

def _bars_html(avg, pace, spin, max_avg, max_pace, max_spin):
    def row(lbl, val, mx, cls):
        pct = 100 * (val / mx) if mx else 0
        return f"""
        <div class='bar-row'>
            <span class='bar-label'>{lbl}</span>
            <div class='bar'><div class='bar-fill {cls}' style='width:{pct:.1f}%;'></div></div>
            <span class='bar-val'>{val:.4f}</span>
        </div>
        """
    return (
        row("Avg Acceleration", avg, max_avg, "avg") +
        row("Pace Accel", pace, max_pace, "pace") +
        row("Spin Accel", spin, max_spin, "spin")
    )

def make_order_cards(order_list, max_avg, max_pace, max_spin):
    """order_list: [(batter, over, avg, pace, spin), ...]"""
    cards = []
    for i, (b, ov, avg, pace, spin) in enumerate(order_list):
        header = "Opener" if i < 2 else f"No. {i+1}"
        bars = _bars_html(avg, pace, spin, max_avg, max_pace, max_spin)
        cards.append(f"""
        <div class='card'>
            <div class='card-header'>{header}</div>
            <div class='name'>{b}</div>
            {bars}
        </div>
        """)
    return "".join(cards)

def make_entry_cards(over_list, max_avg, max_pace, max_spin):
    """over_list: [(over, avg, pace, spin), ...]"""
    cards = []
    for ov, avg, pace, spin in over_list:
        bars = _bars_html(avg, pace, spin, max_avg, max_pace, max_spin)
        cards.append(f"""
        <div class='card'>
            <div class='card-header'>Over {ov}</div>
            {bars}
        </div>
        """)
    return "".join(cards)

def render_cards(html_cards: str, est_rows: int = 4):
    height = min(200 * est_rows + 140, 1200)
    components.html(BASE_CSS + f'<div class="grid">{html_cards}</div>', height=height, scrolling=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {
        'intent_dict':      load_pickle('t20_decay/intents.bin'),
        'p_intent':         load_pickle('t20_decay/paceintents.bin'),
        's_intent':         load_pickle('t20_decay/spinintents.bin'),
        'p_fshot':          load_pickle('t20_decay/pacefshots.bin'),
        's_fshot':          load_pickle('t20_decay/spinfshots.bin'),
        'fshot_dict':       load_pickle('t20_decay/fshots.bin'),
        'gchar':            load_pickle('t20_decay/ground_char.bin'),
        'phase_experience': load_pickle('t20_decay/phase_breakdown.bin'),
        'negdur':           load_pickle('t20_decay/negative_dur.bin'),
        'overwise_dict':    load_pickle('t20_decay/overwise_scores.bin'),
    }

data = load_data()
intent_dict        = data['intent_dict']
p_intent           = data['p_intent']
s_intent           = data['s_intent']
p_fshot            = data['p_fshot']
s_fshot            = data['s_fshot']
fshot_dict         = data['fshot_dict']
gchar              = data['gchar']
phase_experience   = data['phase_experience']
negdur             = data['negdur']
overwise_dict      = data['overwise_dict']

phase_mapping = {i: ("Powerplay (1-6 overs)" if i <= 6 else
                     "Middle (7-11 overs)" if i <= 11 else
                     "Middle (12-16 overs)" if i <= 16 else
                     "Death (17-20 overs)") for i in range(1, 21)}

# ─────────────────────────────────────────────────────────────────────────────
# Core computation helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_top_3_overs(batter, ground_name, num_spinners, num_pacers, n, power, start, power2):
    """Return top n overs as (over, avg, pace_avg, spin_avg)."""
    total_balls = 120
    acc = np.zeros(total_balls)
    pace_arr = np.zeros(total_balls)
    spin_arr = np.zeros(total_balls)

    for s_ball in range(start, total_balls):
        bfaced = 0
        total_intent = 0.0
        pace_total = 0.0
        spin_total = 0.0
        for ball in range(s_ball, total_balls):
            bfaced += 1
            overnum = (ball // 6) + 1
            phase = phase_mapping[overnum]

            paceweight = np.power(gchar[ground_name][overnum - 1] / 100, 1.5)
            spinweight = np.power(1 - gchar[ground_name][overnum - 1] / 100, 1.5)
            spin_prob = spinweight * (num_spinners / (num_spinners + num_pacers)) if (num_spinners + num_pacers) else 0
            pace_prob = paceweight * (num_pacers / (num_spinners + num_pacers)) if (num_spinners + num_pacers) else 0
            total_prob = pace_prob + spin_prob if (pace_prob + spin_prob) != 0 else 1
            pace_prob /= total_prob
            spin_prob /= total_prob

            def get_metric(intent_data, fallback1, key):
                try:
                    if (intent_data['othbatballs'][overnum - 1] == 0 or
                        intent_data['batballs'][overnum - 1] == 0 or
                        intent_data['othbatruns'][overnum - 1] == 0):
                        return fallback1[batter][key]['1-20']
                    return ((intent_data['batruns'][overnum - 1] / intent_data['batballs'][overnum - 1]) /
                            (intent_data['othbatruns'][overnum - 1] / intent_data['othbatballs'][overnum - 1]))
                except Exception:
                    return fallback1[batter][key]['1-20']

            def get_fshot(fshot_data, fallback, key):
                try:
                    if (fshot_data['othbatballs'][overnum - 1] == 0 or
                        fshot_data['batballs'][overnum - 1] == 0 or
                        fshot_data['othbatshots'][overnum - 1] == 0 or
                        fshot_data['batshots'][overnum - 1] == 0):
                        return fallback[batter][key]['1-20']
                    return ((fshot_data['batshots'][overnum - 1] / fshot_data['batballs'][overnum - 1]) /
                            (fshot_data['othbatshots'][overnum - 1] / fshot_data['othbatballs'][overnum - 1]))
                except Exception:
                    return fallback[batter][key]['1-20']

            spin_intent = get_metric(s_intent[batter], intent_dict, 'spin')
            pace_intent = get_metric(p_intent[batter], intent_dict, 'pace')
            spin_fshot  = get_fshot(s_fshot[batter], fshot_dict, 'spin')
            pace_fshot  = get_fshot(p_fshot[batter], fshot_dict, 'pace')

            if bfaced <= negdur[batter]:
                spin_intent = 0
                pace_intent = 0
            if spin_intent < 0.95:
                spin_intent = 0
            if pace_intent < 0.95:
                pace_intent = 0

            phase_weight = phase_experience[batter][phase] / 100
            pace_term = (np.power(pace_intent, power2) * phase_weight * pace_prob / np.power(pace_fshot, power))
            spin_term = (np.power(spin_intent, power2) * phase_weight * spin_prob / np.power(spin_fshot, power))

            total_intent += pace_term + spin_term
            pace_total   += pace_term/(phase_weight * pace_prob)
            spin_total   += spin_term/(phase_weight * spin_prob)

        denom = (total_balls - s_ball) if (total_balls - s_ball) > 0 else 1
        acc[s_ball]      = total_intent / denom
        pace_arr[s_ball] = pace_total   / denom
        spin_arr[s_ball] = spin_total   / denom

    # Aggregate to overs
    over_averages = []
    pace_overs    = []
    spin_overs    = []
    for i in range(0, total_balls, 6):
        over_averages.append(np.mean(acc[i:i+6]))
        pace_overs.append(np.mean(pace_arr[i:i+6]))
        spin_overs.append(np.mean(spin_arr[i:i+6]))

    top_idx = np.argsort(over_averages)[-n:][::-1]
    return [(i+1, over_averages[i], pace_overs[i], spin_overs[i]) for i in top_idx]

def get_optimal_batting_order(batters: dict):
    """batters: {name: [(ov, avg, pace, spin), ...]}"""
    G = nx.Graph()
    for batter, lst in batters.items():
        for ov, avg, pace, spin in lst:
            G.add_edge(batter, f"Over{ov}", weight=avg)

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    batting_order = {}
    total_acc = 0.0
    for a, b in matching:
        if a.startswith("Over"):
            a, b = b, a
        ov = int(b.replace("Over", ""))
        # find tuple
        tup = next(t for t in batters[a] if t[0] == ov)
        avg = tup[1]
        batting_order[a] = tup  # (ov, avg, pace, spin)
        total_acc += avg
    return batting_order, total_acc / max(len(batters), 1)

def get_optimal_batting_order_decay(batters: dict, decay: float = 0.9):
    """Returns dict: name -> (ov, avg, pace, spin, wacc) and avg weighted."""
    batter_over = {b: {ov: (avg, pace, spin) for ov, avg, pace, spin in lst} for b, lst in batters.items()}
    all_overs = sorted({ov for lst in batters.values() for ov, *_ in lst})
    over_to_pos = {ov: i for i, ov in enumerate(all_overs)}

    G = nx.Graph()
    for batter, ov_map in batter_over.items():
        for ov, (avg, pace, spin) in ov_map.items():
            pos = over_to_pos[ov]
            w = (decay ** pos) * avg
            G.add_edge(batter, f"Over{ov}", weight=w)

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")

    batting_order = {}
    total_w = 0.0
    for a, b in matching:
        if a.startswith("Over"):
            a, b = b, a
        ov = int(b.replace("Over", ""))
        avg, pace, spin = batter_over[a][ov]
        pos = over_to_pos[ov]
        wacc = (decay ** pos) * avg
        batting_order[a] = (ov, avg, pace, spin, wacc)
        total_w += avg
    return batting_order, total_w / max(len(matching), 1)

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("T20 Entry Optimization Toolkit")

tab1, tab2, tab3 = st.tabs([
    "Optimal Batting Order",
    "Optimal Entry Point",
    "Scenario-Based Order",
])

common_batters = sorted(set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience))
ground_options = ["Neutral Venue"] + [g for g in gchar if g != "Neutral Venue"]

# Tab 1
with tab1:
    st.subheader("Optimal Batting Order Generator")
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_batters = st.multiselect("Select Batters", common_batters)
        ground = st.selectbox("Select Ground", ground_options, key="tab1_ground")
        spinners = st.slider("Spinners in opposition", 0, 6, 2, key="tab1_spin")
        pacers = st.slider("Pacers in opposition", 0, 6, 4, key="tab1_pace")
        run1 = st.button("Compute Optimal Order", key="run1")
    with col2:
        if run1:
            if not selected_batters:
                st.warning("Select at least one batter to compute order.")
            else:
                batter_map = {b: get_top_3_overs(b, ground, spinners, pacers, 5, 0.5, 0, 1)
                              for b in selected_batters}
                order, avg_score = get_optimal_batting_order(batter_map)
                sorted_order = sorted(order.items(), key=lambda x: x[1][0])

                st.metric("Average Acceleration", f"{avg_score:.4f}")
                order_list = [(b, tup[0], tup[1], tup[2], tup[3]) for b, tup in sorted_order]
                max_avg  = max(x[2] for x in order_list) if order_list else 1
                max_pace = max(x[3] for x in order_list) if order_list else 1
                max_spin = max(x[4] for x in order_list) if order_list else 1
                render_cards(make_order_cards(order_list, max_avg, max_pace, max_spin), est_rows=len(order_list))

                if len(sorted_order) < len(selected_batters):
                    st.markdown("**Note:** Position clashes detected – not optimal to play all of them.")
        else:
            st.info("Configure parameters and click **Compute Optimal Order** to see results.")

# Tab 2
with tab2:
    st.subheader("Optimal Entry Point Calculator")
    col1, col2 = st.columns([1, 3])
    with col1:
        batter = st.selectbox("Select Batter", common_batters, key="tab2_batter")
        ground2 = st.selectbox("Select Ground", ground_options, key="tab2_ground")
        sp2 = st.slider("Spinners in opposition", 0, 6, 2, key="tab2_spin")
        pc2 = st.slider("Pacers in opposition", 0, 6, 4, key="tab2_pace")
        run2 = st.button("Calculate Entry Point", key="run2")
    with col2:
        if run2:
            if sp2 + pc2 == 0:
                st.error("Please select at least one bowler (spinner or pacer).")
            else:
                top_overs = get_top_3_overs(batter, ground2, sp2, pc2, 3, 0.5, 0, 1)
                max_avg  = max(x[1] for x in top_overs) if top_overs else 1
                max_pace = max(x[2] for x in top_overs) if top_overs else 1
                max_spin = max(x[3] for x in top_overs) if top_overs else 1
                render_cards(make_entry_cards(top_overs, max_avg, max_pace, max_spin), est_rows=len(top_overs))
        else:
            st.info("Select parameters and click **Calculate Entry Point** to view top overs.")

# Tab 3
with tab3:
    st.subheader("Scenario-Based Batting Order")
    left, right = st.columns(2)
    with left:
        over_done = st.slider("Overs completed", 0, 20, 6)
        runs_scored = st.number_input("Runs scored so far", 0, 300, 50, step=1)
        chasing = st.checkbox("Chasing target?")
        target = st.number_input("Target score (if chasing)", 1, 300, 180, step=1, disabled=not chasing)
    with right:
        ground3 = st.selectbox("Select Ground", ground_options, key="tab3_ground")
        sp3 = st.slider("Spin overs left", 0, 20, 10, key="spin_sb")
        pc3 = st.slider("Pace overs left", 0, 20, 10, key="pace_sb")
        w_spin = st.slider("Wickets lost to spin", 0, 10, 0, key="spin_w")
        w_pace = st.slider("Wickets lost to pace", 0, 10, 0, key="pace_w")
        avail = st.multiselect("Batters still to come", common_batters, key="bat_left")
        run3 = st.button("Compute Scenario-Based Order", key="run3")

    if run3:
        if over_done >= 20:
            st.error("All overs completed — no batting order to optimise.")
        elif not avail:
            st.warning("Please select at least one batter.")
        elif sp3 + pc3 == 0:
            st.warning("Select at least one spinner or pacer in the opposition.")
        else:
            aggr, rel, decay = aggression_score(
                ground_name   = ground3,
                over_number   = over_done,
                runs_scored   = runs_scored,
                wickets_lost  = w_spin + w_pace,
                ground_over_avg = overwise_dict,
                is_chasing    = chasing,
                target        = target if chasing else None,
            )
            total_wt = aggr + rel if (aggr + rel) != 0 else 1
            aggr = 1.5 * aggr / total_wt
            rel  = 1.5 * rel  / total_wt

            batter_map = {}
            for b in avail:
                top_ovs = get_top_3_overs(
                    batter      = b,
                    ground_name = ground3,
                    num_spinners= sp3 * max(0.5, w_spin),
                    num_pacers  = pc3 * max(0.5, w_pace),
                    n           = 5,
                    power       = rel,
                    start       = 6 * over_done,
                    power2      = aggr
                )
                remaining = [(o, sc, p, s) for o, sc, p, s in top_ovs if o > over_done]
                if remaining:
                    batter_map[b] = remaining

            if not batter_map:
                st.error("No suitable overs left for the chosen batters.")
            else:
                order_sb, avg_acc_sb = get_optimal_batting_order_decay(batter_map, decay)
                order_sorted = sorted(order_sb.items(), key=lambda x: x[1][0])

                st.metric("Scenario Avg Acceleration", f"{avg_acc_sb:.4f}")
                order_list = [(b, tup[0], tup[1], tup[2], tup[3]) for b, tup in order_sorted]
                max_avg  = max(x[2] for x in order_list) if order_list else 1
                max_pace = max(x[3] for x in order_list) if order_list else 1
                max_spin = max(x[4] for x in order_list) if order_list else 1
                render_cards(make_order_cards(order_list, max_avg, max_pace, max_spin), est_rows=len(order_list))
    else:
        st.info("Configure scenario and click **Compute Scenario-Based Order** to view optimized order.")

