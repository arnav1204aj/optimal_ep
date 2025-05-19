import streamlit as st
import numpy as np
import pickle
import networkx as nx
from agg_score import aggression_score
st.set_page_config(page_title="T20 Entry Planner", layout="wide")
# ----------------- Load Data -------------------
@st.cache_data
def load_data():
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    data = {
        # 'absintent' : load_pickle('t20/absintents.bin'),
        'intent_dict': load_pickle('t20/intents.bin'),
        'p_intent': load_pickle('t20/paceintents.bin'),
        's_intent': load_pickle('t20/spinintents.bin'),
        'p_fshot': load_pickle('t20/pacefshots.bin'),
        's_fshot': load_pickle('t20/spinfshots.bin'),
        'fshot_dict': load_pickle('t20/fshots.bin'),
        'gchar': load_pickle('t20/ground_char.bin'),
        'phase_experience': load_pickle('t20/phase_breakdown.bin'),
        'negdur': load_pickle('t20/negative_dur.bin'),
        'overwise_gchar': load_pickle("t20/overwise_scores.bin")
    }
    return data


data = load_data()
# absintent = data['absintent']
intent_dict = data['intent_dict']
p_intent = data['p_intent']
s_intent = data['s_intent']
p_fshot = data['p_fshot']
s_fshot = data['s_fshot']
fshot_dict = data['fshot_dict']
gchar = data['gchar']
phase_experience = data['phase_experience']
negdur = data['negdur']
overwise_dict = data['overwise_gchar']

phase_mapping = {
    i: "Powerplay (1-6 overs)" if i <= 6 else 
       "Middle (7-11 overs)" if i <= 11 else 
       "Middle (12-16 overs)" if i <= 16 else 
       "Death (17-20 overs)" for i in range(1, 21)
}

# ----------------- Helper Functions -------------------
def get_top_3_overs(batter, ground_name, num_spinners, num_pacers, n, power,start,power2):
    acc = np.zeros(120)
    for s_ball in range(start,120):
        bfaced = 0
        intent = 0
        for ball in range(s_ball, 120):
            bfaced += 1
            overnum = (ball // 6) + 1
            phase = phase_mapping[overnum]
            paceweight = np.power(gchar[ground_name][overnum - 1] / 100, 1.5)
            spinweight = np.power(1 - gchar[ground_name][overnum - 1] / 100, 1.5)
            spin_prob = spinweight * (num_spinners / (num_spinners + num_pacers))
            pace_prob = paceweight * (num_pacers / (num_spinners + num_pacers))
            total_prob = pace_prob + spin_prob
            pace_prob /= total_prob
            spin_prob /= total_prob

            # Intent & False Shot Calculation with Fallback
            def get_metric(intent_data, fallback1, key):
                try:
                    if intent_data['othbatballs'][overnum - 1] == 0 or intent_data['batballs'][overnum - 1] == 0 or intent_data['othbatruns'][overnum - 1] == 0:
                        return fallback1[batter][key]['1-20'] #*np.sqrt(fallback2[batter][key]['1-20'])
                    return np.pow(intent_data['batruns'][overnum - 1] / intent_data['batballs'][overnum - 1],1) / \
                           (intent_data['othbatruns'][overnum - 1] / intent_data['othbatballs'][overnum - 1])
                except:
                    return fallback1[batter][key]['1-20'] #*np.sqrt(fallback2[batter][key]['1-20'])

            def get_fshot(fshot_data, fallback, key):
                try:
                    if fshot_data['othbatballs'][overnum - 1] == 0 or fshot_data['batballs'][overnum - 1] == 0 or fshot_data['othbatshots'][overnum - 1] == 0 or fshot_data['batshots'][overnum - 1] == 0:
                        return fallback[batter][key]['1-20']
                    return (fshot_data['batshots'][overnum - 1] / fshot_data['batballs'][overnum - 1]) / \
                           (fshot_data['othbatshots'][overnum - 1] / fshot_data['othbatballs'][overnum - 1])
                except:
                    return fallback[batter][key]['1-20']

            spin_intent = get_metric(s_intent[batter], intent_dict, 'spin')
            pace_intent = get_metric(p_intent[batter], intent_dict, 'pace')
            spin_fshot = get_fshot(s_fshot[batter], fshot_dict, 'spin')
            pace_fshot = get_fshot(p_fshot[batter], fshot_dict, 'pace')

            if bfaced <= negdur[batter]:
                spin_intent = 0
                pace_intent = 0
            if spin_intent < 0.95:
                spin_intent = 0
            if pace_intent < 0.95:
                pace_intent = 0

            phase_weight = phase_experience[batter][phase] / 100
            intent += ((np.pow(pace_intent,power2) * phase_weight * pace_prob / np.pow(pace_fshot,power)) +
                       (np.pow(spin_intent,power2) * phase_weight * spin_prob / np.pow(spin_fshot,power)))

        acc[s_ball] = (intent / (120 - s_ball))
    over_averages = [np.mean(acc[i:i + 6]) for i in range(0, 120, 6)]
    top_3_indices = np.argsort(over_averages)[-n:][::-1]
    return [(i + 1, over_averages[i]) for i in top_3_indices]

def get_optimal_batting_order(batters):
    G = nx.Graph()
    for batter, over_acc_list in batters.items():
        for over, acc in over_acc_list:
            G.add_edge(batter, f"Over{over}", weight=acc)

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    batting_order = {}
    total_acc = 0
    for b, o in matching:
        if b.startswith("Over"):
            b, o = o, b
        over_number = int(o.replace("Over", ""))
        acc_val = dict(batters[b])[over_number]
        batting_order[b] = (over_number, acc_val)
        total_acc += acc_val
    return batting_order, total_acc / len(batters)

import networkx as nx
from itertools import chain

def get_optimal_batting_order_decay(batters: dict, decay: float = 0.9):
    """
    Parameters
    ----------
    batters : dict
        {batter_name: [(over, acceleration), ...], ...}
    decay : float, optional
        Factor in (0, 1]. 1.0 ≡ no decay; 0.9 means each later spot
        is worth 90 % of the previous one.

    Returns
    -------
    batting_order : dict
        {batter_name: (over, raw_acc, weighted_acc)}
    total_weighted_acc : float
        Σ decay**(pos-1) * acc, the maximised objective value
    """
    # --------- helpers -------------------------------------------------------
    # speed-ups for look-ups and consistent over-indexing
    batter_over = {b: dict(lst) for b, lst in batters.items()}
    all_overs      = sorted({ov for lst in batters.values() for ov, _ in lst})
    over_to_pos    = {ov: i for i, ov in enumerate(all_overs)}  # 0-based pos

    # --------- build weighted graph -----------------------------------------
    G = nx.Graph()
    for batter, over_acc in batter_over.items():
        for ov, acc in over_acc.items():
            pos          = over_to_pos[ov]          # batting position (0, 1, …)
            weight       = (decay ** pos) * acc     # decayed weight
            G.add_edge(batter, f"Over{ov}", weight=weight)

    # --------- maximum-weight matching --------------------------------------
    matching = nx.algorithms.matching.max_weight_matching(
        G, maxcardinality=True, weight="weight"
    )

    batting_order = {}
    total_weighted = 0.0
    for a, b in matching:
        # ensure `batter` comes first, `OverX` second
        if a.startswith("Over"):
            a, b = b, a
        ov   = int(b.replace("Over", ""))
        pos  = over_to_pos[ov]
        acc  = batter_over[a][ov]
        wacc = (decay ** pos) * acc

        batting_order[a] = (ov, acc, wacc)
        total_weighted  += acc

    return batting_order, total_weighted/len(matching)



# ----------------- UI -------------------

st.title("T20 Entry Optimization Toolkit")
tab1, tab2, tab3 = st.tabs(
    ["📋 Optimal Batting Order", "📈 Optimal Entry Point", "🎯 Scenario-Based Order"]
)




with tab1:
    st.header("📋 Optimal Batting Order Generator")
    common_batters = set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience)
    selected_batters = st.multiselect("Select Batters", common_batters)
    ground_name = st.selectbox("Select Ground", ["Neutral Venue"] + [g for g in gchar if g != "Neutral Venue"], key="ground1")

    num_spinners = st.slider("Number of spinners in the opposition", 0, 6, 2)
    num_pacers = st.slider("Number of pacers in the opposition", 0, 6, 4)
    if st.button("🔄 Compute Optimal Batting Order"):
        if selected_batters:
            batter_over_map = {batter: get_top_3_overs(batter, ground_name, num_spinners, num_pacers, 5, 0.5,0,1) for batter in selected_batters}
            order, avg_score = get_optimal_batting_order(batter_over_map)
            sorted_order = sorted(order.items(), key=lambda x: x[1][0])
             # Render title and average acceleration
            st.markdown("### ✅ Optimal Batting Order")
            st.markdown(f"#### ⚡ **Average Acceleration:** `{avg_score:.4f}`\n")

        
            table_data = []
            for idx, (batter, (over, score)) in enumerate(sorted_order, 1):
                if idx < 3:
                    label = "Opener"
                else:
                    label = f"In at #{idx}"
                table_data.append([label, "➡️", batter, f"⚡ Accel: {score:.4f}"])

            
            for row in table_data:
                cols = st.columns([1, 0.2, 1, 2])
                cols[0].markdown(row[0])
                cols[1].markdown(row[1])
                cols[2].markdown(f"**{row[2]}**")
                cols[3].markdown(f'<code>{row[3]}</code>', unsafe_allow_html=True)

            if len(sorted_order)<len(selected_batters):
                st.markdown(f"#### ❗Too many position clashes, not optimal to play all of them.")
        else:
                    st.warning("⚠️ Please select at least one batter.")

with tab2:
    st.header("📈 Optimal Entry Point Calculator")
    common_batters = set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience)
    batter = st.selectbox("Select Batter", common_batters, key="batter_entry")
    ground_name = st.selectbox("Select Ground", ["Neutral Venue"] + [g for g in gchar if g != "Neutral Venue"], key="ground2")

    num_spinners = st.slider("Number of spinners in the opposition", 0, 6, 2, key="spin_entry")
    num_pacers = st.slider("Number of pacers in the opposition", 0, 6, 4, key="pace_entry")

    if st.button("📊 Calculate Entry Point"):
       if num_spinners == 0 and num_pacers == 0:
        st.error("❗ Please select at least one bowler (spinner or pacer).")
       else:  
        overs = get_top_3_overs(batter, ground_name, num_spinners, num_pacers,3, 0.5,0,1)
        
        st.markdown("---")
        st.markdown("### Top 3 Optimal Entry Overs")

        # Medal Icons
        medals = ["🥇", "🥈", "🥉"]

        # Style and display each over result
        for rank, (over_num, avg_val) in enumerate(overs):
            st.markdown(f"""
                <div style="
                    background-color:#f0f2f6;
                    padding:12px 20px;
                    margin-bottom:10px;
                    border-left: 6px solid #ff0000;
                    border-radius:8px;
                ">
                    <h5 style="margin:0; color:#222;">{medals[rank]} <b>Over {over_num}</b></h5>
                    <p style="margin:0; color:#444;">
                        Average Acceleration Score: <code style="font-size: 16px;">{avg_val:.4f}</code>
                    </p>
                </div>
            """, unsafe_allow_html=True)

# -------------- Scenario-Based Order tab --------------------------------
with tab3:
    st.header("🎯 Scenario-Based Batting Order")

    # --- match Scenario inputs -----------------------------------------
    col1, col2 = st.columns(2)
    over_done   = col1.slider("Overs completed", 0, 20, 6)
    runs_so_far = col1.number_input("Runs scored so far", 0, 300, 50, step=1)
    

    chasing = col2.checkbox("Chasing target?")
    target  = col2.number_input("Target score (if chasing)", 1, 300, 180, step=1, disabled=not chasing)

    ground_sb = st.selectbox(
        "Select Ground",
        ["Neutral Venue"] + [g for g in overwise_dict if g != "Neutral Venue"],
        key="ground3"
    )

    n_spin_sb  = st.slider("Spin Overs left", 0, 20, 10, key="spin_sb")
    n_pace_sb  = st.slider("Pace Overs left", 0, 20, 10, key="pace_sb")
    spin_strength = st.slider("Wickets lost to spin", 0, 10, 0, key="spin_w")
    pace_strength = st.slider("Wickets lost to pace", 0, 10, 0, key="pace_w")
    # --- pick available batters -----------------------------------------
    common_batters = set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience)
    avail_batters  = st.multiselect("Batters still to come", common_batters, key="bat_left")

    # --- load over-wise averages (only once, cached) --------------------
    # @st.cache_data
    # def load_overwise():
    #     with open("t20/overwise_scores.bin", "rb") as f:
    #         return pickle.load(f)
    # overwise_dict = load_overwise()

    # --------------------------------------------------------------------
    if st.button("🔄 Compute Scenario-Based Order"):
        if over_done >= 20:
            st.error("All overs completed — no batting order to optimise.")
        elif not avail_batters:
            st.warning("Please select at least one batter.")
        elif n_spin_sb + n_pace_sb == 0:
            st.warning("Select at least one spinner or pacer in the opposition.")
        
        else:
            # 1) aggression score (0 ⇒ defensive … 1 ⇒ ultra-aggressive)
            aggr, rel, decay = aggression_score(
                ground_name   = ground_sb,
                over_number   = over_done,
                runs_scored   = runs_so_far,
                wickets_lost  = spin_strength+pace_strength,
                ground_over_avg = overwise_dict,
                is_chasing    = chasing,
                target        = target if chasing else None,
            )
            total_wt = aggr+rel
            aggr = 1.5*aggr/total_wt
            rel = 1.5*rel/total_wt
            # print(aggr,rel,decay)
            # 2) build {batter: top-N overs list}, filtering out overs gone
            batter_map = {}
            for b in avail_batters:
                top_ovs = get_top_3_overs(
                    batter      = b,
                    ground_name = ground_sb,
                    num_spinners= n_spin_sb*max(0.5,spin_strength),
                    num_pacers  = n_pace_sb*max(0.5,pace_strength),
                    n           = 5,          # search window
                    power       = rel,
                    start       = 6*over_done,
                    power2      = aggr                                # ← dynamic ‘power’ factor
                )
                # keep only overs still ahead
                remaining = [(o, sc) for o, sc in top_ovs if o > over_done]
                if remaining:
                    batter_map[b] = remaining

            if not batter_map:
                st.error("No suitable overs left for the chosen batters.")
            else:
                order_sb, avg_acc_sb = get_optimal_batting_order_decay(batter_map,decay)
                order_sorted = sorted(order_sb.items(), key=lambda x: x[1][0])

                st.markdown("### ✅ Scenario-Based Batting Order")
                st.markdown(f"#### ⚡ **Avg Accel:** `{avg_acc_sb:.4f}`")
                st.markdown("---")

                tbl = []
                for idx, (bat, (ov, sc, *_)) in enumerate(order_sorted, 1):
                    label = "Optimal Order" if idx == 1 else f""
                    tbl.append([label,"➡️", bat,  f"⚡ {sc:.4f}"])

                for row in tbl:
                    c = st.columns([1, 0.2, 1, 2])
                    for i, txt in enumerate(row):
                        c[i].markdown(txt)

