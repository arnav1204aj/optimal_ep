def aggression_score(
    ground_name: str,
    over_number: int,
    runs_scored: int,
    wickets_lost: int,
    ground_over_avg: dict,
    is_chasing: bool,
    target:int,
    weight_runs: float = 1,
    weight_wkts: float = 1,
    weight_overs: float = 2,
):
   

    # ── 1) pick the ground (fallback → Neutral Venue) ──────────────────────
    if ground_name not in ground_over_avg:
        ground_name = "Neutral Venue"
    base_over = over_number
    # Clamp over between 1 and 20                                       
    over_number = max(1, min(20, over_number))

    # If that over isn’t in the dictionary (rare), back-fill from the
    # last available over key for that ground.
    if over_number not in ground_over_avg[ground_name]:
        candidate_overs = [o for o in ground_over_avg[ground_name] if o <= over_number]
        if not candidate_overs:                           # nothing earlier? use 1
            candidate_overs = [min(ground_over_avg[ground_name])]
        over_number = max(candidate_overs)

    # ── 2) expected *cumulative* runs & wickets to this over ──────────────
    
    exp_str = ground_over_avg[ground_name][over_number]   # e.g. "48.7-2.3"
    exp_runs, exp_wkts = map(float, exp_str.split("-"))

    if is_chasing:
        total,_ = map(float, ground_over_avg[ground_name][20].split("-"))
        exp_runs = exp_runs*(target/total)
        if base_over==0:
            return 1*(min(2,target/total)),0.5,1
    else:
        if base_over==0:
            return 1,0.5,1    
    # ── 3) RUN component: how far behind / ahead the batting side is ──────
    if exp_runs == 0:                                     # shouldn’t really happen
        run_component = 0.5
    elif runs_scored >= exp_runs:
        run_component = 0.5*min(1.0, (-exp_runs + runs_scored) / runs_scored) 
                                   # no extra urgency
    else:
        # deficit as a fraction of expectation, capped at 1
        run_component = 0.5*min(1.0, (exp_runs - runs_scored) / exp_runs) + 0.5
        

    # ── 4) WICKET component: safety cushion vs expected outs ──────────────
    if exp_wkts == 0:                                     # powerplay start etc.
        wkt_component = 0                               # can attack—no wickets expected yet
    elif wickets_lost >= exp_wkts:
       
        wkt_component = 0.5*(1 - min(1,(wickets_lost-exp_wkts)/wickets_lost))                              # lost more than avg → defend
    else:
        run_component = 1
        # fraction of “wickets in hand” relative to expectation
        wkt_component = 0.5*min(1.0, (exp_wkts - wickets_lost) / exp_wkts) + 0.5
        

    # ── 5) OVERS component: natural escalation to the death overs ─────────
    overs_component = over_number / 20.0                  # 0.05 … 1.00

    # ── 6) combine with optional weights, normalise to [0,1] ─────────────
    weighted_sum = (
        weight_runs  * run_component +
        weight_wkts  * wkt_component +
        weight_overs * overs_component
    )
 
    total_weight = weight_runs + weight_wkts + weight_overs
    score = weighted_sum / total_weight

    # return 1 - round(score, 2)
    return run_component, (1-wkt_component)/2, 1 - overs_component