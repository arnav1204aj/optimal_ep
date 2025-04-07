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

# ---------- User Interface ----------
st.title("📈 Optimal Entry Point Calculator")

# Get common batters from all dictionaries
common_batters = set(intent_dict.keys()) & set(fshot_dict.keys()) & \
                 set(negdur.keys()) & set(phase_experience.keys()) & \
                 set(s_intent.keys()) & set(p_intent.keys()) & \
                 set(s_fshot.keys()) & set(p_fshot.keys())

batter = st.selectbox("Select Batter", sorted(list(common_batters)))
ground_name = st.selectbox("Select Ground", sorted(gchar.keys()))
num_spinners = st.number_input("Number of Spinners in the Opposition", min_value=0, max_value=6, value=2)
num_pacers = st.number_input("Number of Pacers in the Opposition", min_value=0, max_value=6, value=4)

if st.button("Calculate Optimal Entry Point"):
   if num_spinners == 0 and num_pacers == 0:
        st.error("❗ Please select at least one bowler (spinner or pacer).")
   else:      
    acc = np.zeros(120)
    spin_probs = np.zeros(120)
    pace_probs = np.zeros(120)
    for s_ball in range(120):
        bfaced = 0
        intent = 0
        for ball in range(s_ball, 120):
            bfaced += 1
            overnum = (ball // 6) + 1
            phase = phase_mapping[overnum]
            paceweight = gchar[ground_name][overnum - 1] / 100
            spinweight = 1 - paceweight
            paceweight = np.pow(paceweight,1.5)
            spinweight = np.pow(spinweight,1.5)
            spin_prob = spinweight * (num_spinners / (num_spinners + num_pacers))
            pace_prob = paceweight * (num_pacers / (num_spinners + num_pacers))

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

            # False Shot - Spin
            if s_fshot[batter]['othbatballs'][overnum - 1] == 0 or \
               s_fshot[batter]['batballs'][overnum - 1] == 0 or \
               s_fshot[batter]['othbatshots'][overnum - 1] == 0 or \
               s_fshot[batter]['batshots'][overnum - 1] == 0:
                spin_fshot = fshot_dict[batter]['spin']['1-20']
            else:
                spin_fshot = (s_fshot[batter]['batshots'][overnum - 1] / s_fshot[batter]['batballs'][overnum - 1]) / \
                             (s_fshot[batter]['othbatshots'][overnum - 1] / s_fshot[batter]['othbatballs'][overnum - 1])

            # False Shot - Pace
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
                # spin_intent /= 2
                # pace_intent /= 2
                spin_intent=0
                pace_intent=0
            
            intent += ((pace_intent * phase_weight * pace_prob / np.sqrt(pace_fshot)) + \
                      (1.5*spin_intent * phase_weight * spin_prob / np.sqrt(spin_fshot)))
        spin_probs[s_ball] = 1.5*spin_prob
        pace_probs[s_ball] = pace_prob
        acc[s_ball] = intent / (120 - s_ball)

#     max_index = int(np.argmax(acc))
#     max_index = max_index + 1
    
#     st.success(f"📍 Optimal Entry Point: Ball {max_index}  (Over {(max_index-1)//6 + 1})")
       # Average acc per over
    # Average acc per over
    avg_spin_prob  = np.mean(spin_probs)
    avg_pace_prob = np.mean(pace_probs)
    acc = acc/(avg_spin_prob + avg_pace_prob)
    over_averages = [np.mean(acc[i:i+6]) for i in range(0, 120, 6)]

       # Get top 3 overs (1-indexed)
    top_3_indices = np.argsort(over_averages)[-3:][::-1]
    top_3_overs = [(i + 1, over_averages[i]) for i in top_3_indices]

       # Display results with style
    # ---------- Display Top 3 Overs Styled Output ----------
    st.markdown("---")
    st.markdown("### Top 3 Optimal Entry Overs")

       # Medal Icons
    medals = ["🥇", "🥈", "🥉"]

       # Style and display each over result
    for rank, (over_num, avg_val) in enumerate(top_3_overs):
       st.markdown(f"""
        <div style="
            background-color:#f0f2f6;
            padding:12px 20px;
            margin-bottom:10px;
            border-left: 6px solid #4b8bbe;
            border-radius:8px;
        ">
            <h5 style="margin:0; color:#222;">{medals[rank]} <b>Over {over_num}</b></h5>
            <p style="margin:0; color:#444;">
                Average Acceleration Score: <code style="font-size: 16px;">{avg_val:.4f}</code>
            </p>
        </div>
    """, unsafe_allow_html=True)




