import streamlit as st
import numpy as np
import pickle
import networkx as nx
st.set_page_config(page_title="T20 Entry Planner", layout="wide")
# ----------------- Load Data -------------------
@st.cache_data
def load_data():
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    data = {
        'intent_dict': load_pickle('t20/intents.bin'),
        'p_intent': load_pickle('t20/paceintents.bin'),
        's_intent': load_pickle('t20/spinintents.bin'),
        'p_fshot': load_pickle('t20/pacefshots.bin'),
        's_fshot': load_pickle('t20/spinfshots.bin'),
        'fshot_dict': load_pickle('t20/fshots.bin'),
        'gchar': load_pickle('t20/ground_char.bin'),
        'phase_experience': load_pickle('t20/phase_breakdown.bin'),
        'negdur': load_pickle('t20/negative_dur.bin')
    }
    return data


data = load_data()
intent_dict = data['intent_dict']
p_intent = data['p_intent']
s_intent = data['s_intent']
p_fshot = data['p_fshot']
s_fshot = data['s_fshot']
fshot_dict = data['fshot_dict']
gchar = data['gchar']
phase_experience = data['phase_experience']
negdur = data['negdur']

phase_mapping = {
    i: "Powerplay (1-6 overs)" if i <= 6 else 
       "Middle (7-11 overs)" if i <= 11 else 
       "Middle (12-16 overs)" if i <= 16 else 
       "Death (17-20 overs)" for i in range(1, 21)
}

# ----------------- Helper Functions -------------------
def get_top_3_overs(batter, ground_name, num_spinners, num_pacers, n):
    acc = np.zeros(120)
    for s_ball in range(120):
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
            def get_metric(intent_data, fallback, key):
                try:
                    if intent_data['othbatballs'][overnum - 1] == 0 or intent_data['batballs'][overnum - 1] == 0 or intent_data['othbatruns'][overnum - 1] == 0:
                        return fallback[batter][key]['1-20']
                    return (intent_data['batruns'][overnum - 1] / intent_data['batballs'][overnum - 1]) / \
                           (intent_data['othbatruns'][overnum - 1] / intent_data['othbatballs'][overnum - 1])
                except:
                    return fallback[batter][key]['1-20']

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
            intent += ((pace_intent * phase_weight * pace_prob / np.sqrt(pace_fshot)) +
                       (spin_intent * phase_weight * spin_prob / np.sqrt(spin_fshot)))

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

# ----------------- UI -------------------

st.title("T20 Entry Optimization Toolkit")

tab1, tab2 = st.tabs(["📋 Optimal Batting Order", "📈 Optimal Entry Point"])

with tab1:
    st.header("📋 Optimal Batting Order Generator")
    common_batters = sorted(set(intent_dict) & set(fshot_dict) & set(negdur) & set(phase_experience))
    selected_batters = st.multiselect("Select Batters", common_batters)
    ground_name = st.selectbox("Select Ground", sorted(gchar.keys()), key="ground1")
    num_spinners = st.slider("Number of spinners in the opposition", 0, 6, 2)
    num_pacers = st.slider("Number of pacers in the opposition", 0, 6, 4)
    if st.button("🔄 Compute Optimal Batting Order"):
        if selected_batters:
            batter_over_map = {batter: get_top_3_overs(batter, ground_name, num_spinners, num_pacers, 5) for batter in selected_batters}
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
    common_batters = sorted(set(intent_dict) & set(fshot_dict) & set(negdur))
    batter = st.selectbox("Select Batter", common_batters, key="batter_entry")
    ground_name = st.selectbox("Select Ground", sorted(gchar.keys()), key="ground2")
    num_spinners = st.slider("Number of spinners in the opposition", 0, 6, 2, key="spin_entry")
    num_pacers = st.slider("Number of pacers in the opposition", 0, 6, 4, key="pace_entry")

    if st.button("📊 Calculate Entry Point"):
       if num_spinners == 0 and num_pacers == 0:
        st.error("❗ Please select at least one bowler (spinner or pacer).")
       else:  
        overs = get_top_3_overs(batter, ground_name, num_spinners, num_pacers,3)
        
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
