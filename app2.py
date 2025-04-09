import streamlit as st
import numpy as np
import pickle
import networkx as nx

import streamlit as st
import numpy as np
import pickle

path = 't20/intents.bin'
with open(path, "rb") as f:
       intent_dict = pickle.load(f)

path = 't20/paceintents.bin'
with open(path, "rb") as f:
       p_intent = pickle.load(f)

path = 't20/spinintents.bin'
with open(path, "rb") as f:
       s_intent = pickle.load(f)

path = 't20/pacefshots.bin'
with open(path, "rb") as f:
       p_fshot = pickle.load(f)

path = 't20/spinfshots.bin'
with open(path, "rb") as f:
       s_fshot = pickle.load(f)

path = 't20/fshots.bin'
with open(path, "rb") as f:
       fshot_dict = pickle.load(f)


path = 't20/ground_char.bin'
with open(path, "rb") as f:
       gchar = pickle.load(f)

path = 't20/phase_breakdown.bin'
with open(path, "rb") as f:
       phase_experience = pickle.load(f)

path = 't20/negative_dur.bin'
with open(path, "rb") as f:
       negdur = pickle.load(f)

# ---------- Phase Mapping ----------
phase_mapping = {
    i: "Powerplay (1-6 overs)" if i <= 6 else 
       "Middle (7-11 overs)" if i <= 11 else 
       "Middle (12-16 overs)" if i <= 16 else 
       "Death (17-20 overs)" for i in range(1, 21)
}

def get_top_3_overs(batter, ground_name, num_spinners, num_pacers):
    acc = np.zeros(120)
    for s_ball in range(120):
        bfaced = 0
        intent = 0
        for ball in range(s_ball, 120):
            bfaced += 1
            overnum = (ball // 6) + 1
            phase = phase_mapping[overnum]
            paceweight = gchar[ground_name][overnum - 1] / 100
            spinweight = 1 - paceweight
            paceweight = np.pow(paceweight, 1.5)
            spinweight = np.pow(spinweight, 1.5)

            spin_prob = spinweight * (num_spinners / (num_spinners + num_pacers))
            pace_prob = paceweight * (num_pacers / (num_spinners + num_pacers))
            total_prob = pace_prob + spin_prob
            pace_prob = pace_prob / total_prob
            spin_prob = spin_prob / total_prob

            # Spin Intent
            if s_intent[batter]['othbatballs'][overnum - 1] == 0 or \
               s_intent[batter]['batballs'][overnum - 1] == 0 or \
               s_intent[batter]['othbatruns'][overnum - 1] == 0:
                spin_intent = intent_dict[batter]['spin']['1-20']
            else:
                spin_intent = (s_intent[batter]['batruns'][overnum - 1] / s_intent[batter]['batballs'][overnum - 1]) / \
                              (s_intent[batter]['othbatruns'][overnum - 1] / s_intent[batter]['othbatballs'][overnum - 1])

            # Pace Intent
            if p_intent[batter]['othbatballs'][overnum - 1] == 0 or \
               p_intent[batter]['batballs'][overnum - 1] == 0 or \
               p_intent[batter]['othbatruns'][overnum - 1] == 0:
                pace_intent = intent_dict[batter]['pace']['1-20']
            else:
                pace_intent = (p_intent[batter]['batruns'][overnum - 1] / p_intent[batter]['batballs'][overnum - 1]) / \
                              (p_intent[batter]['othbatruns'][overnum - 1] / p_intent[batter]['othbatballs'][overnum - 1])

            # False Shot Probabilities
            if s_fshot[batter]['othbatballs'][overnum - 1] == 0 or \
               s_fshot[batter]['batballs'][overnum - 1] == 0 or \
               s_fshot[batter]['othbatshots'][overnum - 1] == 0 or \
               s_fshot[batter]['batshots'][overnum - 1] == 0:
                spin_fshot = fshot_dict[batter]['spin']['1-20']
            else:
                spin_fshot = (s_fshot[batter]['batshots'][overnum - 1] / s_fshot[batter]['batballs'][overnum - 1]) / \
                             (s_fshot[batter]['othbatshots'][overnum - 1] / s_fshot[batter]['othbatballs'][overnum - 1])

            if p_fshot[batter]['othbatballs'][overnum - 1] == 0 or \
               p_fshot[batter]['batballs'][overnum - 1] == 0 or \
               p_fshot[batter]['othbatshots'][overnum - 1] == 0 or \
               p_fshot[batter]['batshots'][overnum - 1] == 0:
                pace_fshot = fshot_dict[batter]['pace']['1-20']
            else:
                pace_fshot = (p_fshot[batter]['batshots'][overnum - 1] / p_fshot[batter]['batballs'][overnum - 1]) / \
                             (p_fshot[batter]['othbatshots'][overnum - 1] / p_fshot[batter]['othbatballs'][overnum - 1])

            phase_weight = phase_experience[batter][phase] / 100

            if bfaced <= negdur[batter]:
                spin_intent = 0
                pace_intent = 0

            if pace_intent < 0.95:
                pace_intent = 0
            if spin_intent < 0.95:
                spin_intent = 0

            intent += ((pace_intent * phase_weight * pace_prob / np.sqrt(pace_fshot)) +
                       (spin_intent * phase_weight * spin_prob / np.sqrt(spin_fshot)))

        acc[s_ball] = (intent / (120 - s_ball))

    over_averages = [np.mean(acc[i:i + 6]) for i in range(0, 120, 6)]
    top_3_indices = np.argsort(over_averages)[-3:][::-1]
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


# ------- Streamlit UI -------
st.title("📋 Optimal Batting Order Generator")
st.markdown("""
    <div style="position: absolute; top: 0px; right: 10px;">
        <a href="https://optimalentrypoint.streamlit.app/" target="_blank" style="
            text-decoration: none;
            background-color: #4CAF50;
            color: white;
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 14px;
            border: none;
        ">
            Entry Calculator
        </a>
    </div>
""", unsafe_allow_html=True)
common_batters = sorted(set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience))
selected_batters = st.multiselect("Select Batters", common_batters)
ground_name = st.selectbox("Select Ground", sorted(gchar.keys()))
num_spinners = st.slider("Number of Spinners", 0, 6, 2)
num_pacers = st.slider("Number of Pacers", 0, 6, 4)
compute_order = st.button("🔄 Compute Optimal Batting Order", key="compute_button")
if compute_order and selected_batters:
    batter_over_map = {}
    for batter in selected_batters:
        overs = get_top_3_overs(batter, ground_name, num_spinners, num_pacers)
        batter_over_map[batter] = overs

    order, avg_score = get_optimal_batting_order(batter_over_map)
    
        # Sort by entry over
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




elif compute_order:
    st.warning("⚠️ Please select at least one batter.")
