from ortools.sat.python import cp_model
import calendar
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import calendar

def plot_calendar_figure(year, month,
                         num_fdocs,
                         day_weekday,  # dict[d]->weekday(0=Mon..6=Sun)
                         p0, p1, p2,
                         solver):
    """
    Creates a matplotlib figure of the monthly calendar with a 6-row x 7-col grid.
      - Each cell shows:
          * The day number in top-left corner
          * Top line: position #1 doc's name (or single coverage doc if p0)
          * Bottom line: position #2 doc's name (empty if single coverage)
      - A rectangular border is drawn around each cell.
      - Monday=0,... Sunday=6 => columns 0..6
    """
    # 1) Determine days in the month & day-of-week for day=1
    _, days_in_month = calendar.monthrange(year, month)
    first_wd = day_weekday[1]  # 0..6

    # Prepare figure: 6 rows x 7 columns
    fig, axes = plt.subplots(nrows=6, ncols=7, figsize=(14,10))
    fig.suptitle(f"{calendar.month_name[month]} {year}", fontsize=20, y=0.95)

    # Turn off standard axes lines, set x/y range 0..1, so we can draw patches & text
    for r in range(6):
        for c in range(7):
            axes[r,c].set_axis_off()
            axes[r,c].set_xlim(0,1)
            axes[r,c].set_ylim(0,1)

    # Mapping from fdoc ID to name
    doc_name_map = {
        0: "Natthawas",
        1: "Siravich",
        2: "Suphakiat",
        3: "Attapol",
        4: "Thamonwan",
        5: "Ratthapong",
        6: "Jirarat",
        7: "Sretthasith"
    }

    # We'll place day 1 at row=0, col=first_wd
    ROWS, COLS = 6, 7
    current_row = 0
    current_col = first_wd

    for d in range(1, days_in_month + 1):
        # We'll gather which doc is in p0 (single), p1 (pos1), p2 (pos2)
        single_docs = [i for i in range(num_fdocs) if solver.Value(p0[(i,d)]) == 1]
        pos1_docs   = [i for i in range(num_fdocs) if solver.Value(p1[(i,d)]) == 1]
        pos2_docs   = [i for i in range(num_fdocs) if solver.Value(p2[(i,d)]) == 1]

        # Day label in top-left
        axes[current_row, current_col].text(
            0.05, 0.90, f"{d}", ha='left', va='top', fontsize=12
        )

        # If single coverage => top line = that doc's name, bottom line = blank
        # If double coverage => top line = p1 doc's name, bottom line = p2 doc's name
        top_line = ""
        bottom_line = ""
        
        if len(single_docs) == 1:
            # Single coverage
            doc_i = single_docs[0]
            top_line = doc_name_map[doc_i]
            bottom_line = ""
        else:
            # Possibly double coverage
            if len(pos1_docs) == 1:
                top_line = doc_name_map[pos1_docs[0]]
            if len(pos2_docs) == 1:
                bottom_line = doc_name_map[pos2_docs[0]]

        # Place top line at y=0.6, bottom line at y=0.3
        axes[current_row, current_col].text(
            0.5, 0.60, top_line,
            ha='center', va='center', fontsize=12
        )
        axes[current_row, current_col].text(
            0.5, 0.30, bottom_line,
            ha='center', va='center', fontsize=12
        )

        # Draw a rectangular border for the cell
        rect = patches.Rectangle((0,0), 1,1,
                                 fill=False, edgecolor='black')
        axes[current_row, current_col].add_patch(rect)

        # Move to next day
        current_col += 1
        if current_col > 6:
            current_col = 0
            current_row += 1
            if current_row >= ROWS:
                break  # no more cells

    plt.tight_layout()
    plt.show()

def solve_fdoc_schedule_with_p0_p1_p2_and_prints(config):
    """
    Scheduling code with:
      - p0/p1/p2 for single/double coverage.
      - Monday(=0), Tuesday(=1), Sunday(=6) forced to 2-fdoc unless in one_fdoc_day -> 1 or 2.
      - two_fdoc_day => force specific dates to 2 coverage (unless in one_fdoc_day).
      - CCU_fdocs: cannot occupy position #2 on Tuesday.
      - No doc can repeat position #2 on Tuesday more than once.
      - Single coverage day => asterisk (*) in final print.
      - Discouraging shifts on the next 2 days (soft).
      - Workload calculation, iterative Tuesday-limit, etc.
    """
    # --- Config unpack ---
    year = config['year']
    month = config['month']
    num_fdocs = config['num_fdocs']

    vaca_fdocs = config.get('vaca_fdocs', [])
    fdoc_interruptions = config.get('fdoc_interruptions', {})
    fdoc_fixations = config.get('fdoc_fixations', {})
    desired_shifts_per_fdoc = config.get('desired_shifts_per_fdoc', {})
    fdoc_preferences = config.get('fdoc_preferences', {})

    pub_holidays = config.get('pub_holiday', [])
    one_fdoc_day = config.get('one_fdoc_day', [])
    two_fdoc_day = config.get('two_fdoc_day', [])
    
    # The new CCU_fdocs key:
    # -> doc(s) in this list can't do p2 on Tuesday
    CCU_fdocs = config.get('CCU_fdocs', [])

    # Basic date info
    _, days_in_month = calendar.monthrange(year, month)
    day_weekday = {}
    for d in range(1, days_in_month + 1):
        day_weekday[d] = calendar.weekday(year, month, d)  # Monday=0,...Sunday=6

    # 1) base mandatory for Monday=0, Tuesday=1, Sunday=6
    base_mandatory_weekdays = [d for d,wd in day_weekday.items() if wd in [0,1,6]]
    # 2) incorporate two_fdoc_day
    forced_two_fdoc_days = set(two_fdoc_day)
    # unify them
    all_mandatory_two_fdoc_days = set(base_mandatory_weekdays) | forced_two_fdoc_days

    # Identify holiday vs weekday
    weekend_days = [d for d,wd in day_weekday.items() if wd>=5]  # Sat=5,Sun=6
    holiday_days = list(set(weekend_days + pub_holidays))
    weekday_days = [d for d in range(1, days_in_month+1) if d not in holiday_days]

    # We'll do iterative constraints for the "no doc has >2 Tuesday shifts"
    tuesday_days = [d for d,wd in day_weekday.items() if wd==1]
    fdocs_exceeding_tuesday_shifts = set()
    max_iterations = num_fdocs
    iteration = 0

    # Workload weighting
    W0 = 3  # single coverage
    W1 = 2  # pos1
    W2 = 1  # pos2

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        model = cp_model.CpModel()

        # (A) p0,p1,p2
        p0 = {}
        p1 = {}
        p2 = {}
        for i in range(num_fdocs):
            for d in range(1, days_in_month+1):
                p0[(i,d)] = model.NewBoolVar(f'p0_fdoc{i}_day{d}')
                p1[(i,d)] = model.NewBoolVar(f'p1_fdoc{i}_day{d}')
                p2[(i,d)] = model.NewBoolVar(f'p2_fdoc{i}_day{d}')

        # x[i,d] => doc i works day d
        x = {}
        for i in range(num_fdocs):
            for d in range(1, days_in_month+1):
                x[(i,d)] = model.NewBoolVar(f'x_fdoc{i}_day{d}')
                # link x
                model.Add(x[(i,d)] >= p0[(i,d)])
                model.Add(x[(i,d)] >= p1[(i,d)])
                model.Add(x[(i,d)] >= p2[(i,d)])
                model.Add(x[(i,d)] <= p0[(i,d)] + p1[(i,d)] + p2[(i,d)])
                # doc can't occupy multiple positions same day
                model.Add(p0[(i,d)] + p1[(i,d)] + p2[(i,d)] <= 1)

        # Summations
        single_count={}
        pos1_count={}
        pos2_count={}
        for d in range(1, days_in_month+1):
            single_count[d] = model.NewIntVar(0,1, f'single_count_{d}')
            pos1_count[d]   = model.NewIntVar(0,1, f'pos1_count_{d}')
            pos2_count[d]   = model.NewIntVar(0,1, f'pos2_count_{d}')
            model.Add(single_count[d] == sum(p0[(i,d)] for i in range(num_fdocs)))
            model.Add(pos1_count[d]   == sum(p1[(i,d)] for i in range(num_fdocs)))
            model.Add(pos2_count[d]   == sum(p2[(i,d)] for i in range(num_fdocs)))

        # (B) Force coverage
        for d in range(1, days_in_month+1):
            if d in all_mandatory_two_fdoc_days and d not in one_fdoc_day:
                # strictly 2 coverage
                model.Add(single_count[d] == 0)
                model.Add(pos1_count[d]   == 1)
                model.Add(pos2_count[d]   == 1)
            elif d in all_mandatory_two_fdoc_days and d in one_fdoc_day:
                # can be 1 or 2
                day_single = model.NewBoolVar(f'daySingle_{d}')
                model.Add(single_count[d] == 1).OnlyEnforceIf(day_single)
                model.Add(pos1_count[d]   == 0).OnlyEnforceIf(day_single)
                model.Add(pos2_count[d]   == 0).OnlyEnforceIf(day_single)
                model.Add(single_count[d] == 0).OnlyEnforceIf(day_single.Not())
                model.Add(pos1_count[d]   == 1).OnlyEnforceIf(day_single.Not())
                model.Add(pos2_count[d]   == 1).OnlyEnforceIf(day_single.Not())
            else:
                # day can be single=1 or double=2
                day_single = model.NewBoolVar(f'daySingle_{d}')
                model.Add(single_count[d] == 1).OnlyEnforceIf(day_single)
                model.Add(pos1_count[d]   == 0).OnlyEnforceIf(day_single)
                model.Add(pos2_count[d]   == 0).OnlyEnforceIf(day_single)
                model.Add(single_count[d] == 0).OnlyEnforceIf(day_single.Not())
                model.Add(pos1_count[d]   == 1).OnlyEnforceIf(day_single.Not())
                model.Add(pos2_count[d]   == 1).OnlyEnforceIf(day_single.Not())

        # (C) Basic Hard constraints
        # vacation, interruptions
        for i in vaca_fdocs:
            for d in range(1,days_in_month+1):
                model.Add(p0[(i,d)]==0)
                model.Add(p1[(i,d)]==0)
                model.Add(p2[(i,d)]==0)

        for i, intr_days in fdoc_interruptions.items():
            for dd in intr_days:
                if 1<=dd<=days_in_month:
                    model.Add(p0[(i,dd)]==0)
                    model.Add(p1[(i,dd)]==0)
                    model.Add(p2[(i,dd)]==0)

        # fixations => doc i must work day dd
        for i, fix_days in fdoc_fixations.items():
            for dd in fix_days:
                if 1<=dd<=days_in_month:
                    model.Add(x[(i,dd)]==1)

        # no consecutive => x[i,d]+ x[i,d+1] <=1
        for i in range(num_fdocs):
            for d in range(1,days_in_month):
                model.Add(x[(i,d)] + x[(i,d+1)] <=1)

        # min holiday/weekdays
        for i in range(num_fdocs):
            hol_shifts= sum(x[(i,d)] for d in holiday_days)
            wd_shifts=  sum(x[(i,d)] for d in weekday_days)
            model.Add(hol_shifts>=2)
            model.Add(wd_shifts>=4)

        # tuesday-limit => iterative
        for i_doc in fdocs_exceeding_tuesday_shifts:
            model.Add(
                sum(x[(i_doc,d)] for d in tuesday_days) <=2
            )

        # (D) CCU_fdocs constraints
        # 1) CCU cannot occupy p2 on Tuesday => p2[i,tue]=0
        for i_ccu in CCU_fdocs:
            for d in tuesday_days:
                model.Add(p2[(i_ccu,d)]==0)

        # 2) "position2 of tuesday shift must not repeat" => each doc can do p2 on Tuesday at most once
        for i in range(num_fdocs):
            model.Add(
                sum(p2[(i,d)] for d in tuesday_days) <=1
            )

        # (E) Soft constraints
        # 1) desired shifts
        dev_pos={}
        dev_neg={}
        for i in range(num_fdocs):
            desired= desired_shifts_per_fdoc.get(i,0)
            tot_shifts= sum(x[(i,d)] for d in range(1,days_in_month+1))
            dev_pos[i] = model.NewIntVar(0,days_in_month,f'dev_pos_{i}')
            dev_neg[i] = model.NewIntVar(0,days_in_month,f'dev_neg_{i}')
            model.Add(tot_shifts - desired == dev_pos[i] - dev_neg[i])

        # 2) preference
        preference_penalties=[]
        for i, pref_days in fdoc_preferences.items():
            for dd in pref_days:
                if 1<=dd<=days_in_month:
                    pen= model.NewBoolVar(f'pref_pen_{i}_{dd}')
                    model.Add(pen==1).OnlyEnforceIf(x[(i,dd)].Not())
                    model.Add(pen==0).OnlyEnforceIf(x[(i,dd)])
                    preference_penalties.append(pen)

        # 3) discourage day+1, day+2 => soft
        y1_follow={}
        y2_follow={}
        for i in range(num_fdocs):
            for d in range(1, days_in_month+1):
                if d+1<=days_in_month:
                    y1_follow[(i,d)] = model.NewBoolVar(f'y1_follow_{i}_{d}')
                    model.AddBoolAnd([x[(i,d)], x[(i,d+1)]])\
                         .OnlyEnforceIf(y1_follow[(i,d)])
                    model.AddBoolOr([x[(i,d)].Not(), x[(i,d+1)].Not()])\
                         .OnlyEnforceIf(y1_follow[(i,d)].Not())
                if d+2<=days_in_month:
                    y2_follow[(i,d)] = model.NewBoolVar(f'y2_follow_{i}_{d}')
                    model.AddBoolAnd([x[(i,d)], x[(i,d+2)]])\
                         .OnlyEnforceIf(y2_follow[(i,d)])
                    model.AddBoolOr([x[(i,d)].Not(), x[(i,d+2)].Not()])\
                         .OnlyEnforceIf(y2_follow[(i,d)].Not())
        violation_list= list(y1_follow.values())+ list(y2_follow.values())
        total_violations= sum(violation_list)

        # (F) workload => sA= single coverage, sB=pos1, sC=pos2
        sA={}
        sB={}
        sC={}
        for i in range(num_fdocs):
            sA[i]= model.NewIntVar(0, days_in_month, f'sA_{i}')
            sB[i]= model.NewIntVar(0, days_in_month, f'sB_{i}')
            sC[i]= model.NewIntVar(0, days_in_month, f'sC_{i}')
            model.Add(sA[i] == sum(p0[(i,d)] for d in range(1,days_in_month+1)))
            model.Add(sB[i] == sum(p1[(i,d)] for d in range(1,days_in_month+1)))
            model.Add(sC[i] == sum(p2[(i,d)] for d in range(1,days_in_month+1)))

        score={}
        for i in range(num_fdocs):
            score[i]= model.NewIntVar(0,9999,f'score_{i}')
            model.Add(score[i] == (W0*sA[i] + W1*sB[i] + W2*sC[i]))

        score_min = model.NewIntVar(0,9999,'score_min')
        score_max = model.NewIntVar(0,9999,'score_max')
        for i in range(num_fdocs):
            model.Add(score_min <= score[i])
            model.Add(score[i] <= score_max)

        score_range= model.NewIntVar(0,9999,'score_range')
        model.Add(score_range == (score_max - score_min))

        # (G) objective
        weight_dev= 5
        weight_pref=1
        weight_follow=3
        weight_workload=3
        model.Minimize(
            weight_dev* sum(dev_pos[i]+dev_neg[i] for i in range(num_fdocs)) +
            weight_pref* sum(preference_penalties) +
            weight_follow* total_violations +
            weight_workload* score_range
        )

        solver= cp_model.CpSolver()
        solver.parameters.max_time_in_seconds=60
        status= solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # build solution
            fdoc_shift_details= {
                i:{'weekday':[], 'holiday':[], 'shifts':[]}
                for i in range(num_fdocs)
            }
            is_single={}
            for i in range(num_fdocs):
                for d in range(1,days_in_month+1):
                    is_single[(i,d)] = False

            # gather assigned days
            for d in range(1, days_in_month+1):
                for i in range(num_fdocs):
                    if solver.Value(x[(i,d)])==1:
                        fdoc_shift_details[i]['shifts'].append(d)
                        if d in holiday_days:
                            fdoc_shift_details[i]['holiday'].append(d)
                        else:
                            fdoc_shift_details[i]['weekday'].append(d)
                        if solver.Value(p0[(i,d)])==1:
                            is_single[(i,d)] = True

            # check tuesday limit
            tuesday_counts={}
            for i in range(num_fdocs):
                tuesday_counts[i] = sum(1 for dd in fdoc_shift_details[i]['shifts'] 
                                        if dd in tuesday_days)

            overlimit= [i for i,cnt in tuesday_counts.items() if cnt>2]
            if not overlimit:
                print("Solution found with objective:", solver.ObjectiveValue())
                print("-"*60)

                # SHIFT ASSIGNMENT
                print("\nShift Assignments (with Position #1, #2):")
                print("------------------------------------------------------------")
                for d in range(1,days_in_month+1):
                    p0_list= [i for i in range(num_fdocs) if solver.Value(p0[(i,d)])==1]
                    p1_list= [i for i in range(num_fdocs) if solver.Value(p1[(i,d)])==1]
                    p2_list= [i for i in range(num_fdocs) if solver.Value(p2[(i,d)])==1]
                    assigned_count= len(p0_list)+len(p1_list)+len(p2_list)

                    day_type= "HOL" if d in holiday_days else "WD"
                    abbr= calendar.day_abbr[day_weekday[d]]
                    doc_list=[]
                    for i_doc in p0_list:
                        doc_list.append(f"F{i_doc:02d}")
                    for i_doc in p1_list:
                        doc_list.append(f"F{i_doc:02d}")
                    for i_doc in p2_list:
                        doc_list.append(f"F{i_doc:02d}")
                    doc_str= ", ".join(doc_list)
                    print(f"{d} {abbr} [{day_type}] : {assigned_count} fdocs [ {doc_str} ]")

                # SHIFT ASSIGNMENT (Config)
                print("\nShift Assignments (Config-Friendly Format):")
                for i in range(num_fdocs):
                    sorted_shifts= sorted(fdoc_shift_details[i]['shifts'])
                    s_str= ", ".join(str(dd) for dd in sorted_shifts)
                    print(f"{i}: [{s_str}],")

                # TUESDAY SHIFT COUNT
                print("\nTuesday Shift Counts:")
                for i in range(num_fdocs):
                    print(f"F{i:02d}: {tuesday_counts[i]} Tuesday shift(s)")

                # 1-Fdoc vs 2-Fdoc
                one_fdoc_count= [0]*num_fdocs
                two_fdoc_count= [0]*num_fdocs
                for d in range(1,days_in_month+1):
                    if sum(solver.Value(p0[(i,d)]) for i in range(num_fdocs))==1:
                        i_doc= [k for k in range(num_fdocs) if solver.Value(p0[(k,d)])==1][0]
                        one_fdoc_count[i_doc]+=1
                    else:
                        # double coverage => p1+p2
                        for i_doc in range(num_fdocs):
                            if solver.Value(p1[(i_doc,d)])==1 or solver.Value(p2[(i_doc,d)])==1:
                                two_fdoc_count[i_doc]+=1

                print("\n1-Fdoc vs. 2-Fdoc Shifts:")
                for i in range(num_fdocs):
                    print(f"F{i:02d}: 1-Fdoc={one_fdoc_count[i]},  2-Fdoc={two_fdoc_count[i]}")

                # Detailed shift with "*"
                print("\nDetailed Shift Counts per Fdoc:")
                for i in range(num_fdocs):
                    wdays= sorted(fdoc_shift_details[i]['weekday'])
                    hdays= sorted(fdoc_shift_details[i]['holiday'])
                    wdays_str=[]
                    for dd in wdays:
                        if is_single[(i,dd)]:
                            wdays_str.append(f"{dd}*")
                        else:
                            wdays_str.append(f"{dd}")
                    hdays_str=[]
                    for dd in hdays:
                        if is_single[(i,dd)]:
                            hdays_str.append(f"{dd}*")
                        else:
                            hdays_str.append(f"{dd}")
                    total_days= sorted(wdays+hdays)
                    total_str=[]
                    for dd in total_days:
                        if is_single[(i,dd)]:
                            total_str.append(f"{dd}*")
                        else:
                            total_str.append(f"{dd}")
                    
                    print(f"F{i:02d}:")
                    print(f"  Weekday Shifts = {len(wdays)} [{', '.join(wdays_str)}]")
                    print(f"  Holiday Shifts = {len(hdays)} [{', '.join(hdays_str)}]")
                    print(f"  Total Shifts   = [{', '.join(total_str)}]")

                # Workload
                print("\nWorkload Info:")
                score_vals=[]
                for i in range(num_fdocs):
                    sA_val= solver.Value(sA[i])
                    sB_val= solver.Value(sB[i])
                    sC_val= solver.Value(sC[i])
                    sc= solver.Value(score[i])
                    score_vals.append(sc)
                    print(f"F{i:02d} => singleDays={sA_val}, pos1Days={sB_val}, pos2Days={sC_val}, Score={sc}")

                print(f"Score range: {min(score_vals)} {max(score_vals)}")
                print("\nDone.")

                # Plot calendar figure
                plot_calendar_figure(year, month,
                                     num_fdocs,
                                     day_weekday,
                                     p0, p1, p2,
                                     solver)
                break
            else:
                print("Docs exceeding 2 Tuesday shifts =>", overlimit)
                for i_doc in overlimit:
                    fdocs_exceeding_tuesday_shifts.add(i_doc)
        else:
            print("No feasible solution found.")
            return

if __name__=="__main__":
    # Example usage
    config = {
        'year':2025,
        'month':2,
        'num_fdocs':8,

        # example ccu
        'CCU_fdocs': [0],  # say doc #2 is the CCU doc => can't do p2 on Tuesday
        'vaca_fdocs': [],

        'fdoc_interruptions': {
            0:[28],
            1:[13,14,15,16,23,28],
            2:[11,14,15,16,25],
            3:[1,9,22,23],
            4:[9,14,15,16,26,27],
            5:[1,2,6,13,14,15,16,19],
            6:[2,6,14,15,16,20],
            7:[6,13,14,15,16],
        },
        'fdoc_fixations': {
0: [1, 5, 11, 15, 19, 24],
1: [3, 9, 12, 18, 21, 26],
2: [3, 6, 9, 13, 18, 22],
3: [2, 6, 14, 16, 19, 25],
4: [2, 7, 10, 17, 22, 25],
5: [5, 8, 11, 17, 20, 23],
6: [4, 8, 12, 21, 24, 27],
7: [1, 4, 10, 20, 23, 28],
        },
        'desired_shifts_per_fdoc': {
            0:3,1:6,2:6,3:6,4:6,5:6,6:6,7:6,
        },
        'fdoc_preferences': {
            # example usage
        },
        'pub_holiday':[12],
        'one_fdoc_day':[14,15,16],
        'two_fdoc_day':[5,19,22],
    }

    solve_fdoc_schedule_with_p0_p1_p2_and_prints(config)
