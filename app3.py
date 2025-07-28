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
    background-size: contain;
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
    background-size: contain;
    background-position: center;
    border-radius: 8px;
    margin-bottom: 12px;
}}
.player-img,
.entry-player-img {{
    background-repeat: no-repeat;
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
.bar-fill.avg   {{ background: {THEME_RED}; }}
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
        'intent_dict':       lp('t20_decay/intents.bin'),
        'p_intent':          lp('t20_decay/paceintents.bin'),
        's_intent':          lp('t20_decay/spinintents.bin'),
        'p_fshot':           lp('t20_decay/pacefshots.bin'),
        's_fshot':           lp('t20_decay/spinfshots.bin'),
        'fshot_dict':        lp('t20_decay/fshots.bin'),
        'gchar':             lp('t20_decay/ground_char.bin'),
        'phase_experience':  lp('t20_decay/phase_breakdown.bin'),
        'negdur':            lp('t20_decay/negative_dur.bin'),
        'overwise_dict':     lp('t20_decay/overwise_scores.bin'),
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

# Reverse mapping for phase names to over ranges
reverse_phase_mapping = {
    "Powerplay (1-6 overs)": (1, 6),
    "Middle (7-11 overs)": (7, 11),
    "Middle (12-16 overs)": (12, 16),
    "Death (17-20 overs)": (17, 20)
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
            row("Pace Int-Rel",        pace, max_pace, "pace") +
            row("Spin Int-Rel",        spin, max_spin, "spin"))


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
        <div class='meta'>Opt. Entry Phase: {dominant_phase}</div>
      </div>
    </div>
    """

def make_suggested_player_cards(suggested_players_data, max_avg, max_pace, max_spin):
    cards = []
    for rank, (name, avg, pace, spin, optimal_over, required_phase) in enumerate(suggested_players_data):
        header = ("Top Suggestion" if  rank == 0
                  else f"Suggestion {rank+1}")
        img_div = f"<div class='player-img' style=\"background-image:url('{image_map.get(name,'')}');\"></div>"
        cards.append(f"""
        <div class='card'>
          {img_div}
          <div class='card-header'>{header}</div>
          <div class='name'>{name}</div>
          {_bars_html(avg, pace, spin, max_avg, max_pace, max_spin)}
        </div>""")
    return "".join(cards)


def render_cards(html, est_rows=4):
    # height = min(200 * est_rows + 140, 1200)
    height = 200 * est_rows + 140
    components.html(BASE_CSS + f'<div class="grid">{html}</div>',
                     height=height, scrolling=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_top_3_overs(batter, ground_name, num_spinners, num_pacers, n, power, start, power2,end=120):
    """Return top n overs as (over, avg, pace_avg, spin_avg)."""
    total_balls = 120
    acc = np.zeros(total_balls)
    pace_arr = np.zeros(total_balls)
    spin_arr = np.zeros(total_balls)


    for s_ball in range(start, end):
        bfaced = 0
        total_intent = 0.0
        pace_total = 0.0
        spin_total = 0.0
        for ball in range(s_ball, total_balls):
            bfaced += 1
            overnum = (ball // 6) + 1
            # Ensure overnum is within valid range (1-20)
            if overnum > 20: # Prevent index errors for remaining balls if start is very late
                continue
            phase = phase_mapping[overnum]


            # Ensure ground_name and overnum-1 are valid indices
            if ground_name not in gchar or len(gchar[ground_name]) <= (overnum - 1):
                # Fallback or skip if data is missing for this ground/over
                # For simplicity, we'll assign default weights if data is missing.
                # In a real scenario, you might want to log this or use a more robust fallback.
                paceweight = 0.5 # Default to balanced if ground data is missing
                spinweight = 0.5
            else:
                paceweight = np.power(gchar[ground_name][overnum - 1] / 100, 1.5)
                spinweight = np.power(1 - gchar[ground_name][overnum - 1] / 100, 1.5)

            spin_prob = spinweight * (num_spinners / (num_spinners + num_pacers)) if (num_spinners + num_pacers) else 0
            pace_prob = paceweight * (num_pacers / (num_spinners + num_pacers)) if (num_spinners + num_pacers) else 0
            total_prob = pace_prob + spin_prob if (pace_prob + spin_prob) != 0 else 1
            # Avoid division by zero if total_prob is 0
            if total_prob > 0:
                pace_prob /= total_prob
                spin_prob /= total_prob


            def get_metric(intent_data, fallback1, key):
                try:
                    # Check for valid data before division
                    if (batter not in intent_data or
                        'othbatballs' not in intent_data[batter] or
                        len(intent_data[batter]['othbatballs']) <= (overnum - 1) or
                        intent_data[batter]['othbatballs'][overnum - 1] == 0 or
                        intent_data[batter]['batballs'][overnum - 1] == 0 or
                        intent_data[batter]['othbatruns'][overnum - 1] == 0):
                        return fallback1[batter][key]['1-20']
                    return ((intent_data[batter]['batruns'][overnum - 1] / intent_data[batter]['batballs'][overnum - 1]) /
                            (intent_data[batter]['othbatruns'][overnum - 1] / intent_data[batter]['othbatballs'][overnum - 1]))
                except Exception:
                    # Fallback if any error occurs
                    return fallback1[batter][key]['1-20']


            def get_fshot(fshot_data, fallback, key):
                try:
                    # Check for valid data before division
                    if (batter not in fshot_data or
                        'othbatballs' not in fshot_data[batter] or
                        len(fshot_data[batter]['othbatballs']) <= (overnum - 1) or
                        fshot_data[batter]['othbatballs'][overnum - 1] == 0 or
                        fshot_data[batter]['batballs'][overnum - 1] == 0 or
                        fshot_data[batter]['othbatshots'][overnum - 1] == 0 or
                        fshot_data[batter]['batshots'][overnum - 1] == 0):
                        return fallback[batter][key]['1-20']
                    return ((fshot_data[batter]['batshots'][overnum - 1] / fshot_data[batter]['batballs'][overnum - 1]) /
                            (fshot_data[batter]['othbatshots'][overnum - 1] / fshot_data[batter]['othbatballs'][overnum - 1]))
                except Exception:
                    # Fallback if any error occurs
                    return fallback[batter][key]['1-20']

            if batter not in s_intent or batter not in p_intent or batter not in s_fshot or batter not in p_fshot:
                # If batter data is missing for intents/fshots, skip this ball calculation
                # This could happen if `common_batters` includes players with incomplete data
                continue

            spin_intent = get_metric(s_intent, intent_dict, 'spin')
            pace_intent = get_metric(p_intent, intent_dict, 'pace')
            spin_fshot  = get_fshot(s_fshot, fshot_dict, 'spin')
            pace_fshot  = get_fshot(p_fshot, fshot_dict, 'pace')


            if bfaced <= negdur.get(batter, 0): # Use .get for robustness
                spin_intent = 0
                pace_intent = 0
            if spin_intent < 0.95:
                spin_intent = 0
            if pace_intent < 0.95:
                pace_intent = 0

            # Ensure phase_experience has data for the batter and phase
            if batter not in phase_experience or phase not in phase_experience[batter]:
                 phase_weight = 1 # Default to 1 if phase experience data is missing
            else:
                 phase_weight = phase_experience[batter][phase] / 100

            # Avoid division by zero for pace_prob and spin_prob in pace_term/spin_term if they are 0
            pace_term = 0
            if phase_weight * pace_prob != 0:
                pace_term = (np.power(pace_intent, power2) * phase_weight * pace_prob / np.power(pace_fshot, power))
            
            spin_term = 0
            if phase_weight * spin_prob != 0:
                spin_term = (np.power(spin_intent, power2) * phase_weight * spin_prob / np.power(spin_fshot, power))


            total_intent += pace_term + spin_term
            
            if (phase_weight * pace_prob) != 0:
                pace_total   += pace_term/(phase_weight * pace_prob)
            if (phase_weight * spin_prob) != 0:
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

    # Handle cases where all values might be NaN or zero for empty ranges if data was missing
    if not over_averages:
        return []

    # Get top_n indices, ensuring n doesn't exceed available overs
    num_available_overs = len(over_averages)
    n_to_select = min(n, num_available_overs)

    # Use argsort to get indices that would sort the array, then pick the last 'n' (highest values)
    # If over_averages contains NaNs, argsort will place them at the end or beginning depending on numpy version.
    # It's better to filter out NaNs or handle them appropriately before sorting if they are expected.
    
    # Filter out potential NaNs before sorting if necessary, though mean() should return 0 for empty arrays or valid number.
    # Let's assume valid numbers or 0.
    
    top_idx = np.argsort(over_averages)[-n_to_select:][::-1] # Get indices of top n values, then reverse for descending order
    return [(i+1, over_averages[i], pace_overs[i], spin_overs[i]) for i in top_idx]


def get_optimal_batting_order(batters: dict):
    """batters: {name: [(ov, avg, pace, spin), ...]}"""
    G = nx.Graph()
    for batter, lst in batters.items():
        # Add edges only if lst is not empty
        if lst:
            for ov, avg, pace, spin in lst:
                G.add_edge(batter, f"Over{ov}", weight=avg)

    # Ensure maxcardinality is set appropriately. If some batters might not match an over,
    # max_weight_matching might include them without a match, or you need to handle that.
    # Here, we assume a match is always possible for selected batters.
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    
    batting_order = {}
    total_acc = 0.0
    matched_batters = set() # Track batters that successfully got a match

    # Iterate through the matching, which gives (node1, node2) pairs
    for u, v in matching:
        # Determine which one is the batter and which is the over
        batter_node = u if not u.startswith("Over") else v
        over_node = v if v.startswith("Over") else u

        ov = int(over_node.replace("Over", ""))
        
        # Retrieve the original tuple from the batters dictionary
        # We need to find the specific tuple for this batter and over
        found_tup = None
        if batter_node in batters:
            for tup in batters[batter_node]:
                if tup[0] == ov:
                    found_tup = tup
                    break
        
        if found_tup:
            batting_order[batter_node] = found_tup  # (ov, avg, pace, spin)
            total_acc += found_tup[1] # Add the acceleration value
            matched_batters.add(batter_node)

    # Calculate average acceleration based on successfully matched batters
    return batting_order, total_acc / max(len(matched_batters), 1)


def get_optimal_batting_order_decay(batters: dict, decay: float = 0.9):
    """Returns dict: name -> (ov, avg, pace, spin, wacc) and avg weighted."""
    batter_over = {b: {ov: (avg, pace, spin) for ov, avg, pace, spin in lst} for b, lst in batters.items()}
    all_overs = sorted(list({ov for lst in batters.values() for ov, *_ in lst})) # Ensure unique and sorted
    over_to_pos = {ov: i for i, ov in enumerate(all_overs)}


    G = nx.Graph()
    for batter, ov_map in batter_over.items():
        for ov, (avg, pace, spin) in ov_map.items():
            pos = over_to_pos[ov]
            w = (decay ** pos) * avg
            G.add_edge(batter, f"Over{ov}", weight=w)

    # Use 'weight' attribute to find max weight matching
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True, weight="weight")


    batting_order = {}
    total_w = 0.0
    matched_pairs_count = 0 # To count how many actual matches were made for averaging

    for u, v in matching:
        # Determine which one is the batter and which is the over
        batter_node = u if not u.startswith("Over") else v
        over_node = v if v.startswith("Over") else u

        ov = int(over_node.replace("Over", ""))
        
        if batter_node in batter_over and ov in batter_over[batter_node]:
            avg, pace, spin = batter_over[batter_node][ov]
            pos = over_to_pos[ov]
            wacc = (decay ** pos) * avg
            batting_order[batter_node] = (ov, avg, pace, spin, wacc)
            total_w += wacc # Sum weighted acceleration
            matched_pairs_count += 1
            
    return batting_order, total_w / max(matched_pairs_count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("T20 Entry Optimization Toolkit")


tab1, tab2, tab3, tab4, tab5 = st.tabs([ # Added tab5 for "Info" and tab4 for "Suggest a Batter"
    "Optimal Batting Order",
    "Optimal Entry Point",
    "Scenario-Based Order",
    "Suggest Best Batters", # New tab
    "Info"
])


common_batters = sorted(list(set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience)))
grounds = ["Neutral Venue"] + [g for g in gchar if g != "Neutral Venue"]


# --- TAB 1: Optimal Batting Order ---
with tab1:
    st.subheader("Optimal Batting Order Generator")
    c1, c2 = st.columns([1, 3])
    with c1:
        sel     = st.multiselect("Select Batters", common_batters, key="t1_sel")
        g1      = st.selectbox("Select Ground", grounds, key="t1_g")
        spn     = st.slider("Spinners in Opposition", 0, 6, 2,   key="t1_sp")
        pac     = st.slider("Pacers in Opposition",   0, 6, 4,   key="t1_pc")
        compute = st.button("Compute Optimal Order", key="t1_go")
    with c2:
        if compute:
            if not sel:
                st.warning("Select at least one batter.")
            else:
                bm = {}
                for b in sel:
                    # Ensure batter has data before calling get_top_3_overs
                    if b in intent_dict and b in fshot_dict and b in negdur and b in phase_experience:
                        bm[b] = get_top_3_overs(b, g1, spn, pac, 5, 0.5, 0, 1)
                    else:
                        st.warning(f"Data missing for batter: {b}. Skipping.")

                if not bm:
                    st.error("No valid batter data found for computation.")
                else:
                    optimal_over = {
                        batter: max(data, key=lambda x: x[1])[0]
                        for batter, data in bm.items()
                        if data # ignore empty lists
                    }

                    # 2) Count how many batters got assigned to each over
                    freq = Counter(optimal_over.values())

                    # 3) Sum the counts for any over where count > 1
                    clashes = sum(count for count in freq.values() if count > 1)
                    order, avg = get_optimal_batting_order(bm)
                    so    = sorted(order.items(), key=lambda x: x[1][0])
                    ol    = [(b, t[0], t[1], t[2], t[3]) for b, t in so]
                    ma    = max((x[2] for x in ol), default=1)
                    mp    = max((x[3] for x in ol), default=1)
                    ms    = max((x[4] for x in ol), default=1)

                    col1, col2 = st.columns(2)
                    col1.metric("Average Acceleration", f"{avg:.4f}")
                    col2.metric("Positional Clashes", str(clashes))

                    render_cards(make_order_cards(ol, ma, mp, ms, False), est_rows=len(ol))
        else:
            st.info("Configure and compute.")


# --- TAB 2: Optimal Entry Point ---
with tab2:
    st.subheader("Optimal Entry Point Calculator")
    c1, c2 = st.columns([1, 3])
    with c1:
        batter = st.selectbox("Select Batter", common_batters, key="e_bat")
        ground2 = st.selectbox("Select Ground", grounds, key="e_gnd")
        sp2 = st.slider("Spinners in Opposition", 0, 6, 2, key="e_sp")
        pc2 = st.slider("Pacers in Opposition",   0, 6, 4, key="e_pc")
        go2 = st.button("Calculate Entry Point", key="e_go")
    with c2:
        if go2:
            if sp2 + pc2 == 0:
                st.error("Select at least one bowler in opposition.")
            elif batter not in intent_dict or batter not in fshot_dict or batter not in negdur or batter not in phase_experience:
                st.error(f"Data missing for batter: {batter}. Cannot calculate entry point.")
            else:
                overs = get_top_3_overs(batter, ground2, sp2, pc2, 3, 0.5, 0, 1)

                if not overs:
                    st.warning(f"Could not determine optimal overs for {batter} with given parameters.")
                else:
                    # determine dominant phase
                    phases = [phase_mapping[o] for o, _, _, _ in overs]
                    cnt = Counter(phases)
                    maxc = max(cnt.values())
                    cands = [ph for ph, c in cnt.items() if c == maxc]
                    phase_avg = {
                        ph: np.mean([avg for o, avg, _, _ in overs if phase_mapping[o] == ph])
                        for ph in cands
                    }
                    dominant = sorted(phase_avg.items(), key=lambda x: -x[1])[0][0]


                    # build combined HTML
                    ma = max(avg for _, avg, _, _ in overs) if overs else 1
                    mp = max(p for *_, p in overs) if overs else 1
                    ms = max(s for *_, _, s in overs) if overs else 1

                    combined = (
                        make_entry_player_card(batter, dominant) +
                        make_entry_cards(overs, ma, mp, ms)
                    )
                    render_cards(combined, est_rows=1)
        else:
            st.info("Configure parameters and click Calculate Entry Point.")


# --- TAB 3: Scenario-Based Order ---
with tab3:
    st.subheader("Scenario-Based Order")
    L, R = st.columns(2)
    with L:
        od    = st.slider("Overs completed", 0,20,6, key="t3_od")
        e_s   = st.number_input("Run rate vs spin",0.0,100.0,8.33,step=0.01, key="t3_es")
        e_p   = st.number_input("Run rate vs pace",0.0,100.0,8.33,step=0.01, key="t3_ep")
        ground3 = st.selectbox("Select Ground", grounds, key="t3_g")
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
            try:
                ag, rel, dec = aggression_score(
                    ground_name     = ground3,
                    over_number     = od,
                    runs_scored     = rs,
                    wickets_lost    = ws + wp,
                    ground_over_avg = overwise_dict,
                    is_chasing      = chase,
                    target          = tgt if chase else None
                )
            except Exception as e:
                st.error(f"Error calculating aggression score: {e}. Please check inputs.")
                ag, rel, dec = 1.0, 1.0, 0.9 # Fallback values
                
            tot = ag + rel or 1
            ag, rel = 1.5*ag/tot, 1.5*rel/tot
            
            # Handle potential division by zero for tw and te
            tw = ws + wp if (ws + wp) > 0 else 1
            te = e_s + e_p if (e_s + e_p) > 0 else 1
            
            sw, pw = (ws/tw)+(e_p/te), (wp/tw)+(e_s/te)


            bm = {}
            for b in avail:
                # Ensure batter has data before calling get_top_3_overs
                if b in intent_dict and b in fshot_dict and b in negdur and b in phase_experience:
                    ov = get_top_3_overs(b, ground3,
                                         s3*max(0.5, sw),
                                         p3*max(0.5, pw),
                                         5, rel, 6*od, ag)
                    rem = [(o,sc,pa,sp) for o,sc,pa,sp in ov if o > od]
                    if rem:
                        bm[b] = rem
                else:
                    st.warning(f"Data missing for batter: {b}. Skipping.")


            if not bm:
                st.error("No overs left or no valid batter data for computation.")
            else:
                optimal_over = {
                    batter: max(data, key=lambda x: x[1])[0]
                    for batter, data in bm.items()
                    if data # ignore empty lists
                }

                # 2) Count how many batters got assigned to each over
                freq = Counter(optimal_over.values())

                # 3) Sum the counts for any over where count > 1
                clashes = sum(count for count in freq.values() if count > 1)
                order3, avg3 = get_optimal_batting_order_decay(bm, dec)
                so3 = sorted(order3.items(), key=lambda x: x[1][0])
                ol3 = [(b, t[0], t[1], t[2], t[3]) for b, t in so3]
                ma3 = max((x[2] for x in ol3), default=1)
                mp3 = max((x[3] for x in ol3), default=1)
                ms3 = max((x[4] for x in ol3), default=1)

                col1, col2 = st.columns(2)
                col1.metric("Average Acceleration", f"{avg3:.4f}")
                col2.metric("Positional Clashes", str(clashes))

                render_cards(make_order_cards(ol3, ma3, mp3, ms3, True), est_rows=len(ol3))
    else:
        st.info("Configure and compute scenario.")

# --- TAB 4: Suggest a Batter ---
with tab4:
    st.subheader("Batter Suggestion")
    c1, c2 = st.columns([1, 3])
    with c1:
        ground4 = st.selectbox("Select Ground", grounds, key="t4_gnd")
        required_phase = st.selectbox(
            "Required Entry Phase",
            list(reverse_phase_mapping.keys()),
            key="t4_phase"
        )
        sp4 = st.slider("Spinners in Opposition", 0, 6, 2, key="t4_sp")
        pc4 = st.slider("Pacers in Opposition",   0, 6, 4, key="t4_pc")
        candidate_batters = st.multiselect("Candidate Batters", common_batters, key="t4_cands")
        suggest_button = st.button("Suggest Batters", key="t4_suggest")

    with c2:
        start = (reverse_phase_mapping[required_phase][0]-1)*6
        end = (reverse_phase_mapping[required_phase][1])*6
        if suggest_button:
            if not candidate_batters:
                st.warning("Please select at least one candidate batter.")
            elif sp4 + pc4 == 0:
                st.warning("Please specify at least one bowler in opposition (spinners or pacers).")
            else:
                suggested_results = []
                phase_start_over, phase_end_over = reverse_phase_mapping[required_phase]

                for batter_name in candidate_batters:
                    if batter_name not in intent_dict or batter_name not in fshot_dict or batter_name not in negdur or batter_name not in phase_experience:
                        st.warning(f"Data missing for batter: {batter_name}. Skipping.")
                        continue

                    # Get top 5 overs for the batter under current conditions (enough to check all phases)
                    batter_optimal_overs = get_top_3_overs(batter_name, ground4, sp4, pc4, 1, 1, start, 1,end)

                    best_in_phase = None
                    # Iterate through the optimal overs to find the best one within the required phase
                    for ov, avg, pace, spin in batter_optimal_overs:
                        if phase_start_over <= ov <= phase_end_over:
                            if best_in_phase is None or avg > best_in_phase[1]: # Compare by avg acceleration
                                best_in_phase = (ov, avg, pace, spin)
                    
                    if best_in_phase:
                        suggested_results.append((batter_name, best_in_phase[1], best_in_phase[2], best_in_phase[3], best_in_phase[0], required_phase))

                if not suggested_results:
                    st.info("No suitable batters found for the specified phase and conditions.")
                else:
                    # Sort by average acceleration in descending order
                    suggested_results.sort(key=lambda x: x[1], reverse=True)

                    # Determine max values for bar rendering
                    max_avg = max(x[1] for x in suggested_results) if suggested_results else 1
                    max_pace = max(x[2] for x in suggested_results) if suggested_results else 1
                    max_spin = max(x[3] for x in suggested_results) if suggested_results else 1

                    rendered_html = make_suggested_player_cards(suggested_results, max_avg, max_pace, max_spin)
                    render_cards(rendered_html, est_rows=len(suggested_results))
        else:
            st.info("Configure parameters and click 'Suggest Batters'.")


# --- TAB 5: Info --- (renamed from tab4)
with tab5:
    st.subheader("About This App")
    st.markdown(
        """
        **Developer:** Arnav Jain   
        **Contact:** [arnav1204aj@gmail.com](mailto:arnav1204aj@gmail.com)   

        
        You can find the detailed metrics and methodology behind this app on my Substack:   
        [arnavj.substack.com](https://arnavj.substack.com/)   


        Read the full breakdown of *The Batting Order Toolkit* here:   
        [The Batting Order Toolkit](https://arnavj.substack.com/p/the-batting-order-toolkit)
        """
    )