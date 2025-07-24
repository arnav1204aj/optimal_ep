import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import networkx as nx
from collections import Counter
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

/* Entry‑player‑card layout */
.entry-card {{
  display: flex;
  align-items: center;
  gap: 12px;
}}
.entry-player-img {{
  width: 80px;
  height: 80px;
  background-color: #f5f5f7;
  background-size: cover;
  background-position: center;
  border-radius: 8px;
}}

/* Common */
.card-header {{
  font-size: 1.05rem;
  font-weight: 600;
  color: {THEME_RED};
  margin-bottom: 10px;
  letter-spacing: 0.2px;
}}
.player-img {{
  width: 100%;
  height: 150px;
  background-color: #f5f5f7;
  background-size: cover;
  background-position: center;
  border-radius: 8px;
  margin-bottom: 12px;
}}
.name {{
  font-weight: 600;
  font-size: 1rem;
  margin: 0 0 8px 0;
}}
.meta {{
  opacity: .80;
  font-size: 0.9rem;
  margin-bottom: 4px;
}}

/* Bars */
.bar-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0 6px 0;
}}
.bar-label {{ width: 120px; font-size: 0.8rem; color: #444; }}
.bar-val   {{ font-size: 0.8rem; color: #111; width: 52px; text-align: right; }}
.bar {{ flex: 1; background: #e6e6e6; border-radius: 6px; height: 8px; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 6px; }}
.bar-fill.avg  {{ background: {THEME_RED}; }}
.bar-fill.pace {{ background: #004c97; }}
.bar-fill.spin {{ background: #2ca02c; }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    def lp(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = {
        'intent_dict':      lp('t20_decay/intents.bin'),
        'p_intent':         lp('t20_decay/paceintents.bin'),
        's_intent':         lp('t20_decay/spinintents.bin'),
        'p_fshot':          lp('t20_decay/pacefshots.bin'),
        's_fshot':          lp('t20_decay/spinfshots.bin'),
        'fshot_dict':       lp('t20_decay/fshots.bin'),
        'gchar':            lp('t20_decay/ground_char.bin'),
        'phase_experience': lp('t20_decay/phase_breakdown.bin'),
        'negdur':           lp('t20_decay/negative_dur.bin'),
        'overwise_dict':    lp('t20_decay/overwise_scores.bin'),
    }
    players_df = pd.read_csv("players.csv", usecols=["fullname", "image_path"])
    data["image_map"] = players_df.set_index("fullname")["image_path"].to_dict()
    return data

data = load_data()
intent_dict       = data['intent_dict']
p_intent          = data['p_intent']
s_intent          = data['s_intent']
p_fshot           = data['p_fshot']
s_fshot           = data['s_fshot']
fshot_dict        = data['fshot_dict']
gchar             = data['gchar']
phase_experience  = data['phase_experience']
negdur            = data['negdur']
overwise_dict     = data['overwise_dict']
image_map         = data['image_map']

# ─────────────────────────────────────────────────────────────────────────────
# PHASE MAPPING
# ─────────────────────────────────────────────────────────────────────────────
phase_mapping = {
    i: ("Powerplay (1-6 overs)" if i <= 6 else
        "Middle (7-11 overs)"   if i <= 11 else
        "Middle (12-16 overs)"  if i <= 16 else
        "Death (17-20 overs)")
    for i in range(1, 21)
}

# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _bars_html(avg, pace, spin, max_avg, max_pace, max_spin):
    def row(lbl, val, mx, cls):
        pct = 100 * (val / mx) if mx else 0
        return f"""
        <div class='bar-row'>
          <span class='bar-label'>{lbl}</span>
          <div class='bar'><div class='bar-fill {cls}' style='width:{pct:.1f}%;'></div></div>
          <span class='bar-val'>{val:.4f}</span>
        </div>"""
    return (row("Avg Acceleration", avg, max_avg, "avg") +
            row("Pace Int-Rel",      pace, max_pace, "pace") +
            row("Spin Int-Rel",      spin, max_spin, "spin"))

def make_order_cards(order_list, max_avg, max_pace, max_spin, is_scenario):
    cards = []
    for i, (b, ov, avg, pace, spin) in enumerate(order_list):
        header = ("Next In" if is_scenario and i == 0
                  else "Opener" if not is_scenario and i < 2
                  else f"No. {i+1}")
        img_div = f"<div class='player-img' style=\"background-image:url('{image_map.get(b,'')}');\"></div>"
        cards.append(f"""
        <div class='card'>
          {img_div}
          <div class='card-header'>{header}</div>
          <div class='name'>{b}</div>
          {_bars_html(avg, pace, spin, max_avg, max_pace, max_spin)}
        </div>""")
    return "".join(cards)

def make_entry_cards(over_list, max_avg, max_pace, max_spin):
    cards = []
    for ov, avg, pace, spin in over_list:
        cards.append(f"""
        <div class='card'>
          <div class='card-header'>Over {ov}</div>
          {_bars_html(avg, pace, spin, max_avg, max_pace, max_spin)}
        </div>""")
    return "".join(cards)

def make_entry_player_card(name, dominant_phase):
    return f"""
    <div class='card entry-card'>
      <div class='entry-player-img' style="background-image:url('{image_map.get(name,'')}');"></div>
      <div>
        <div class='name'>{name}</div>
        <div class='meta'>Dominant Phase: {dominant_phase}</div>
      </div>
    </div>
    """

def render_cards(html, est_rows=4):
    height = min(200 * est_rows + 140, 1200)
    components.html(BASE_CSS + f'<div class="grid">{html}</div>',
                    height=height, scrolling=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION HELPERS
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
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("T20 Entry Optimization Toolkit")

tab1, tab2, tab3 = st.tabs([
    "Optimal Batting Order",
    "Optimal Entry Point",
    "Scenario-Based Order",
])

common_batters = sorted(set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience))
grounds = ["Neutral Venue"] + [g for g in gchar if g != "Neutral Venue"]

# --- TAB 1: Optimal Batting Order (unchanged) ---
# -- TAB 1: Optimal Batting Order --
with tab1:
    st.subheader("Optimal Batting Order Generator")
    c1,c2 = st.columns([1,3])
    with c1:
        sel     = st.multiselect("Select Batters", common_batters, key="t1_sel")
        g1      = st.selectbox("Select Ground", grounds, key="t1_g")
        spn     = st.slider("Spinners opp", 0,6,2, key="t1_sp")
        pac     = st.slider("Pacers opp",   0,6,4, key="t1_pc")
        compute = st.button("Compute Optimal Order", key="t1_go")
    with c2:
        if compute:
            if not sel:
                st.warning("Select at least one batter.")
            else:
                bm = {b: get_top_3_overs(b, g1, spn, pac, 5, 0.5, 0, 1) for b in sel}
                order, avg = get_optimal_batting_order(bm)
                so    = sorted(order.items(), key=lambda x: x[1][0])
                ol    = [(b,t[0],t[1],t[2],t[3]) for b,t in so]
                ma    = max((x[2] for x in ol), default=1)
                mp    = max((x[3] for x in ol), default=1)
                ms    = max((x[4] for x in ol), default=1)
                st.metric("Average Acceleration", f"{avg:.4f}")
                render_cards(make_order_cards(ol, ma, mp, ms, False), est_rows=len(ol))
        else:
            st.info("Configure and compute.")
# --- TAB 2: Optimal Entry Point with combined grid ---
with tab2:
    st.subheader("Optimal Entry Point Calculator")
    c1, c2 = st.columns([1, 3])
    with c1:
        batter = st.selectbox("Select Batter", common_batters, key="e_bat")
        ground2 = st.selectbox("Select Ground", grounds, key="e_gnd")
        sp2 = st.slider("Spinners opp", 0, 6, 2, key="e_sp")
        pc2 = st.slider("Pacers opp",   0, 6, 4, key="e_pc")
        go2 = st.button("Calculate Entry Point", key="e_go")
    with c2:
        if go2:
            if sp2 + pc2 == 0:
                st.error("Select at least one batter.")
            else:
                overs = get_top_3_overs(batter, ground2, sp2, pc2, 3, 0.5, 0, 1)

                # dominant phase
                phases = [phase_mapping[o] for o,_,_,_ in overs]
                cnt = Counter(phases)
                maxc = max(cnt.values())
                cands = [ph for ph,c in cnt.items() if c==maxc]
                phase_avg = {
                  ph: np.mean([avg for o,avg,_,_ in overs if phase_mapping[o]==ph])
                  for ph in cands
                }
                dominant = sorted(phase_avg.items(), key=lambda x:-x[1])[0][0]

                # build HTML
                ma = max(avg for _,avg,_,_ in overs) if overs else 1
                mp = max(p   for *_,p   in overs) if overs else 1
                ms = max(s   for *_,_,s in overs) if overs else 1

                combined = (
                  make_entry_player_card(batter, dominant) +
                  make_entry_cards(overs, ma, mp, ms)
                )
                render_cards(combined, est_rows=1)
        else:
            st.info("Configure parameters and click Calculate Entry Point.")

with tab3:
    st.subheader("Scenario-Based Order")
    L,R = st.columns(2)
    with L:
        od    = st.slider("Overs completed", 0,20,6, key="t3_od")
        e_s   = st.number_input("Run rate vs spin",0.0,100.0,8.33,step=0.01, key="t3_es")
        e_p   = st.number_input("Run rate vs pace",0.0,100.0,8.33,step=0.01, key="t3_ep")
        ground3 = st.selectbox("Select Ground", grounds, key="tab3_ground")
        chase = st.checkbox("Chasing?", key="t3_ch")
        tgt   = st.number_input("Target",1,300,180, disabled=not chase, key="t3_tg")
    with R:
        rs    = st.number_input("Runs so far", 0,300,50, key="t3_rs")
        ws    = st.slider("Wickets lost (spin)",0,10,0, key="t3_ws")
        wp    = st.slider("Wickets lost (pace)",0,10,0, key="t3_wp")
        s3    = st.slider("Spin overs left", 0,20,10, key="t3_s3")
        p3    = st.slider("Pace overs left", 0,20,10, key="t3_p3")
        avail = st.multiselect("Batters left", common_batters, key="t3_av")
        go3   = st.button("Compute Scenario-Based Order", key="t3_go")

    if go3:
        if od >= 20:
            st.error("All overs done.")
        elif not avail:
            st.warning("Select at least one batter.")
        elif s3 + p3 == 0:
            st.warning("Select at least one bowler.")
        else:
            ag,rel,dec = aggression_score(
                ground_name   = ground3,
                over_number   = od,
                runs_scored   = rs,
                wickets_lost  = ws + wp,
                ground_over_avg = overwise_dict,
                is_chasing    = chase,
                target        = tgt if chase else None
            )
            tot = ag + rel or 1
            ag,rel = 1.5*ag/tot, 1.5*rel/tot
            tw = ws + wp or 1
            te = e_s + e_p or 1
            sw = (ws/tw) + (e_p/te)
            pw = (wp/tw) + (e_s/te)

            bm = {}
            for b in avail:
                ov = get_top_3_overs(b, grounds[0],
                                     s3*max(0.5, sw),
                                     p3*max(0.5, pw),
                                     5, rel, 6*od, ag)
                rem = [(o,sc,pa,sp) for o,sc,pa,sp in ov if o > od]
                if rem:
                    bm[b] = rem

            if not bm:
                st.error("No overs left.")
            else:
                order3, avg3 = get_optimal_batting_order_decay(bm, dec)
                so3 = sorted(order3.items(), key=lambda x: x[1][0])
                ol3 = [(b,t[0],t[1],t[2],t[3]) for b,t in so3]
                ma3 = max((x[2] for x in ol3), default=1)
                mp3 = max((x[3] for x in ol3), default=1)
                ms3 = max((x[4] for x in ol3), default=1)

                st.metric("Scenario Avg Acceleration", f"{avg3:.4f}")
                render_cards(make_order_cards(ol3, ma3, mp3, ms3, True), est_rows=len(ol3))
    else:
        st.info("Configure and compute scenario.")

