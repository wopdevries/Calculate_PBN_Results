
# streamlit program to display deal statistics from a PBN file.
# Invoke from system prompt using: streamlit run CalculatePBNResults_Streamlit.py

import streamlit as st
import pathlib
import fsspec
import pandas as pd # only used for __version__ for now. might need for plotting later as pandas plotting support is better than polars.
import polars as pl
import duckdb
import pickle
from collections import defaultdict
from datetime import datetime, timezone
import sys

import endplay # for __version__
from endplay.parsers import pbn
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import par, calc_all_tables
from endplay.dealer import generate_deals

sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
import streamlitlib


def create_df_from_pbn(boards):
    # this may look weird but there's all sorts of opportunities to error otherwise. polars was giving rust errors when creating a DataFrame from a list of objects.
    cols_d = {k:[getattr(b,k) for b in boards] for k in vars(boards[0])} # create dict with keys of column names and values are rows
    simple_data_type_cols = ['board_num', '_dealer', '_vul', 'claimed']
    simples = pl.DataFrame([cols_d[col] for col in simple_data_type_cols],schema=simple_data_type_cols)
    infos = pl.DataFrame(cols_d['info']) # automatically expands dict keys to columns
    contracts = pl.DataFrame(cols_d['_contract'],schema=['_contract']) # 
    plays = pl.DataFrame(map(str,cols_d['play']),schema=['play'])
    # plays = pl.DataFrame(cols_d['play'],schema=['play']) # polars.exceptions.ShapeError: data does not match the number of columns
    # [PenaltyBid(penalty=<Penalty.passed: 1>, alertable=False, announcement=None), ContractBid(denom=<Denom.clubs: 3>, level=1, alertable=False, announcement=None), PenaltyBid(penalty=<Penalty.doubled: 2>, alertable=False, announcement=None), ContractBid(denom=<Denom.spades: 0>, level=1, alertable=False, announcement=None)]
    auctions = pl.DataFrame(map(str,cols_d['auction']),schema=['auction'])
    # auctions = pl.DataFrame(cols_d['auction'],schema=['auction']) # polars.exceptions.ShapeError: data does not match the number of columns
    deals = pl.DataFrame(cols_d['deal'],schema=['deal'])
    df = pl.concat([simples,infos,contracts,plays,auctions,deals],how='horizontal')
    return df


def calculate_ddtricks_par_scores(df, progress=None):

    NSEW_direction_order = [0, 2, 1, 3]
    SHDCN_suit_order = [3, 2, 1, 0, 4]
    # Calculate double dummy and par
    deals = df['deal'] # using original deal object
    batch_size = 40
    t_t = []
    tables = []
    for b in range(0, len(deals), batch_size):
        if progress:
            percent_complete = int(b*100/len(deals))
            progress.progress(percent_complete,f"{percent_complete}%: {b} of {len(deals)} double dummies calculated.")
        batch_tables = calc_all_tables(deals[b:min(b + batch_size, len(deals))])
        tables.extend(batch_tables)
        batch_t_t = (tt._data.resTable for tt in batch_tables)
        t_t.extend(batch_t_t)
    if progress:
        progress.progress(100,f"100%: {len(deals)} of {len(deals)} double dummies calculated.")

    assert len(deals) == len(t_t) == len(tables)

    # Display a few hands and double dummy tables
    # max_display = 4
    # for ii, (dd, sd, tt) in enumerate(zip(deals, t_t, tables)):
    #     if ii < max_display:
    #         st.write(f"Deal: {ii + 1}")
    #         st.write(dd)
    #         st.write(tt)
    #         st.write(tuple(tuple(sd[suit][direction] for suit in SHDCN_suit_order) for direction in NSEW_direction_order))

    # Create dataframe of par scores using double dummy
    pars = [par(tt, Vul.find(v), 0) for tt, v in zip(tables, df['Vul'])]  # middle arg is board number (if int) otherwise enum vul. Must use Vul.find(v) because uncorrelated to board number.
    par_scores_ns = [parlist.score for parlist in pars]
    par_scores_ew = [-score for score in par_scores_ns]
    par_contracts = [', '.join([str(contract.level) + 'SHDCN'[int(contract.denom)] + contract.declarer.abbr + contract.penalty.abbr + ('' if contract.result == 0 else '+'+str(contract.result) if contract.result > 0 else str(contract.result)) for contract in parlist]) for parlist in pars]
    par_df = pl.DataFrame({'ParScore_NS': par_scores_ns, 'ParScore_EW': par_scores_ew, 'ParContract': par_contracts},orient='row')

    # Create dataframe of double dummy tricks per direction and suit
    DDTricks_rows = [[sd[suit][direction] for direction in NSEW_direction_order for suit in SHDCN_suit_order] for sd in t_t]
    DDTricks_df = pl.DataFrame(DDTricks_rows, schema=['_'.join(['DDTricks', d, s]) for d in 'NSEW' for s in 'CDHSN'],orient='row')

    def Tricks_To_Score(sd):
        return [Contract(level=level, denom=suit, declarer=direction, penalty=Penalty.passed if sd[suit][direction] - 6 - level >= 0 else Penalty.doubled, result=sd[suit][direction] - 6 - level).score(0) for direction in NSEW_direction_order for suit in SHDCN_suit_order for level in range(1, 8)]

    dd_score_rows = [Tricks_To_Score(sd) for sd in t_t]
    dd_score_df = pl.DataFrame(dd_score_rows, schema=['_'.join(['DDScore', str(l) + s, d]) for d in 'NSEW' for s in 'CDHSN' for l in range(1, 8)],orient='row')

    return DDTricks_df, par_df, dd_score_df


# todo: could save a couple seconds by creating dict of deals
def calc_double_dummy_deals(deals, batch_size=40):
    t_t = []
    tables = []
    for b in range(0,len(deals),batch_size):
        batch_tables = calc_all_tables(deals[b:min(b+batch_size,len(deals))])
        tables.extend(batch_tables)
        batch_t_t = (tt._data.resTable for tt in batch_tables)
        t_t.extend(batch_t_t)
    assert len(t_t) == len(tables)
    return deals, t_t, tables


def constraints(deal):
    return True


def generate_single_dummy_deals(predeal_string, produce, env=dict(), max_attempts=1000000, seed=None, show_progress=True, strict=True, swapping=0):
    
    predeal = Deal(predeal_string)

    deals_t = generate_deals(
        constraints,
        predeal=predeal,
        swapping=swapping,
        show_progress=show_progress,
        produce=produce,
        seed=seed,
        max_attempts=max_attempts,
        env=env,
        strict=strict
        )

    deals = tuple(deals_t) # create a tuple before interop memory goes wonky
    
    return calc_double_dummy_deals(deals)


def calculate_single_dummy_probabilities(deal, produce=100):

    ns_ew_rows = {}
    for ns_ew in ['NS','EW']:
        s = deal[2:].split()
        if ns_ew == 'NS':
            s[1] = '...'
            s[3] = '...'
        else:
            s[0] = '...'
            s[2] = '...'
        predeal_string = 'N:'+' '.join(s)
        #print_to_log(f"predeal:{predeal_string}")

        d_t, t_t, tables = generate_single_dummy_deals(predeal_string, produce, show_progress=False)

        rows = []
        max_display = 4 # pprint only the first n generated deals
        direction_order = [0,2,1,3] # NSEW order
        suit_order = [3,2,1,0,4] # SHDCN order?
        for ii,(dd,sd,tt) in enumerate(zip(d_t,t_t,tables)):
            # if ii < max_display:
                # print_to_log(f"Deal:{ii+1} Fixed:{ns_ew} Generated:{ii+1}/{produce}")
                # dd.pprint()
                # print_to_log()
                # tt.pprint()
                # print_to_log()
            nswe_flat_l = [sd[suit][direction] for direction in direction_order for suit in suit_order]
            rows.append([dd.to_pbn()]+nswe_flat_l)

        dd_df = pl.DataFrame(rows,schema=['Deal']+[d+s for d in 'NSEW' for s in 'CDHSN'],orient='row')
        fill_values = {i:0 for i in range(14)} # fill values for missing tricks
        for d in 'NSEW':
            for s in 'SHDCN':
                # todo: convert this line from pandas to polars
                #ns_ew_rows[(ns_ew,d,s)] = dd_df[d+s].to_pandas().value_counts(normalize=True).reindex(range(14), fill_value=0).tolist() # ['Fixed_Direction','Direction_Declarer','Suit']+['SD_Prob_Take_'+str(n) for n in range(14)]
                vc = {ds:p for ds,p in dd_df[d+s].value_counts(normalize=True).rows()}
                index = {i:0.0 for i in range(14)} # fill values for missing probs
                ns_ew_rows[(ns_ew,d,s)] = list((index|vc).values())
    return ns_ew_rows


# def append_single_dummy_results(pbns,sd_cache_d,produce=100):
#     for pbn in pbns:
#         if pbn not in sd_cache_d:
#             sd_cache_d[pbn] = calculate_single_dummy_probabilities(pbn, produce) # all combinations of declarer pair directI. ion, declarer direciton, suit, tricks taken
#     return sd_cache_d


# takes 1000 seconds for 100 sd calcs, or 10 sd calcs per second.
def calculate_sd_probs(df, sd_productions=100, progress=None):
    sd_cache_d = {}
    deals = df['Deal']
    for i,deal in enumerate(deals):
        if progress:
            percent_complete = int(i*100/len(deals))
            progress.progress(percent_complete,f"{percent_complete}%: {i} of {len(deals)} single dummies calculated using {sd_productions} samples")
        # st.write(f"{percent_complete}%: {i} of {len(deals)} boards. deal:{deal}")
        if deal not in sd_cache_d:
            sd_cache_d[deal] = calculate_single_dummy_probabilities(deal, sd_productions) # all combinations of declarer pair direction, declarer direciton, suit, tricks taken
    if progress:
        progress.progress(100,f"100%: {len(deals)} of {len(deals)} single dummies calculated.")


    # calculate single dummy trick taking probability distribution
    sd_probs_d = defaultdict(list)
    for deal in deals:
        v = sd_cache_d[deal]
        # st.write(pbn,v)
        for (pair_direction,declarer_direction,suit),probs in v.items():
            for i,t in enumerate(probs):
                sd_probs_d['_'.join(['Probs',pair_direction,declarer_direction,suit,str(i)])].append(t)
    # st.write(sd_probs_d)
    sd_probs_df = pl.DataFrame(sd_probs_d,orient='row')
    return sd_cache_d, sd_probs_df


# calculate dict of contract result scores
def calculate_scores():
    SHDCN_suit_order = [3, 2, 1, 0, 4]

    scores_d = {}
    for suit in SHDCN_suit_order:
        for level in range(1,8): # contract level
            for tricks in range(14):
                result = tricks-6-level
                scores_d[(level,'SHDCN'[suit],tricks,False)] = Contract(level=level,denom=suit,declarer=0,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(False)
                scores_d[(level,'SHDCN'[suit],tricks,True)] = Contract(level=level,denom=suit,declarer=0,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(True)

    # create score dataframe from dict
    all_scores_d = defaultdict(list)
    for suit in 'SHDCN':
        for level in range(1,8):
            for i in range(14):
                all_scores_d['_'.join(['Score',str(level)+suit])].append([scores_d[(level,suit,i,False)],scores_d[(level,suit,i,True)]])
    # st.write(all_scores_d)
    scores_df = pl.DataFrame(all_scores_d,orient='row')
    #scores_df.index.name = 'Taken'
    return all_scores_d, scores_df


def calculate_sd_expected_values(df,sd_cache_d,scores_d):
    # create dict of expected values (probability * score)
    exp_d = defaultdict(list)
    deal_vul = zip(df['Deal'],df['Vul'])
    for deal,vul in deal_vul:
        #st.write(deal,vul)
        for (pair_direction,declarer_direction,suit),probs in sd_cache_d[deal].items():
            is_declarer_vul = vul == 'Both' or (vul != 'None' and declarer_direction in vul)
            #st.write(pair_direction,declarer_direction,suit,probs,is_declarer_vul)
            for level in range(1,8):
                #st.write(scores_d['_'.join(['Score',str(level)+suit])][is_declarer_vul])
                exp_d['_'.join(['Exp',pair_direction,declarer_direction,suit,str(level)])].append(sum([prob*score[is_declarer_vul] for prob,score in zip(probs,scores_d['_'.join(['Score',str(level)+suit])])]))
            #st.write(exp_d)
    #st.write(exp_d)
    sd_exp_df = pl.DataFrame(exp_d,orient='row')
    return sd_exp_df


# create columns containing the 1) the name of the column having the max expected value. 2) max expected value 3) contract having the max expected value.
def create_best_contracts(df):
    besties = []
    for d in df.to_dicts():
        exp_tuples = tuple([(v,k) for k,v in d.items()])
        ex_tuples_sorted = sorted(exp_tuples,reverse=True)
        best_contract_tuple = ex_tuples_sorted[0]
        best_contract_split = best_contract_tuple[1].split('_') # split column name into parts
        best_contract = best_contract_split[4]+best_contract_split[3]+best_contract_split[2]
        besties.append([best_contract_tuple[1],best_contract_tuple[0],best_contract_tuple[0] if best_contract_tuple[1][-5] in ['N','S'] else -best_contract_tuple[0],best_contract])
    return besties


def calculate_best_contracts(sd_exp_df):
    sd_best_contract_l = create_best_contracts(sd_exp_df) #sd_exp_df.to_pandas().apply(create_best_contracts,axis='columns')
    sd_best_contract_df = pl.DataFrame(sd_best_contract_l,schema=['ExpMaxScore_Col','Exp_Max','ExpMaxScore_NS','BestContract'],orient='row')
    return sd_best_contract_df


def convert_contract_to_contract(df):
    return df['Contract'].str.to_uppercase().str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C').str.replace('NT','N')


# None is used instead of pl.Null because pl.Null becomes 'Null' string in pl.String columns. Not sure what's going on but the solution is to use None.
def convert_contract_to_declarer(df):
    return [None if c == 'PASS' else c[2] for c in df['Contract']] # extract declarer from contract


def convert_declarer_to_DeclarerName(df):
    return [None if d is None else df[d][i] for i,d in enumerate(df['Declarer'])] # extract declarer name using declarer direction as the lookup key


def convert_contract_to_result(df):
    return [None if c == 'PASS' else 0 if c[-1] in ['=','0'] else int(c[-1]) if c[-2] == '+' else -int(c[-1]) for c in df['Contract']] # create result from contract


def convert_contract_to_tricks(df):
    return [None if c == 'PASS' else int(c[0])+6+r for c,r in zip(df['Contract'],df['Result'])] # create tricks from contract and result


def convert_contract_to_DDTricks(df):
    return [None if c == 'PASS' else df['_'.join(['DDTricks',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_score_to_score(df):
    scores = []
    for d in df.to_dicts():
        score_split = d['Score'].split()
        assert len(score_split) == 2, f"score_split:{score_split}"
        assert score_split[0] in ['NS','EW'], f"score_split:{score_split[0]}"
        assert score_split[1][0] == '-' or str.isdigit(score_split[1][0]), f"score_split:{score_split[1]}"
        score_split_direction = score_split[0]
        score_split_value = score_split[1]
        score_value = -int(score_split_value) if score_split_value[0] == '-' else int(score_split_value)
        scores.append(score_value if score_split_direction == 'NS' else -score_value)
    return scores


def create_augmented_df(df):
    df = df.clone()
    df = df.rename({'North':'N','East':'E','South':'S','West':'W'}) # todo: is this really better?

    df = df.with_columns(
        pl.Series('Contract',convert_contract_to_contract(df),pl.String,strict=False), # can have nulls or Strings
    )
    df = df.with_columns(
        pl.Series('Declarer',convert_contract_to_declarer(df),pl.String,strict=False), # can have nulls or Strings
    )
    df = df.with_columns(
        pl.Series('DeclarerName',convert_declarer_to_DeclarerName(df),pl.String,strict=False), # can have nulls or Strings
        pl.Series('Result',convert_contract_to_result(df),pl.Int8,strict=False), # can have nulls or Int8
    )
    df = df.with_columns(
        pl.Series('Tricks',convert_contract_to_tricks(df),pl.UInt8,strict=False), # can have nulls or UInt8
        pl.Series('DDTricks',convert_contract_to_DDTricks(df),pl.UInt8,strict=False), # can have nulls or UInt8
        pl.Series('Score_NS',convert_score_to_score(df),pl.Int16),
    )
    df = df.with_columns(
        pl.Series('ParScore_Diff_NS',(df['Score_NS']-df['ParScore_NS']),pl.Int16),
        # needs to have .cast(pl.Int8) because left and right are both UInt8 which goofs up the subtraction.
        pl.Series('DDTricks_Diff',(df['Tricks'].cast(pl.Int8)-df['DDTricks'].cast(pl.Int8)),pl.Int8,strict=False), # can have nulls or Int8
        pl.Series('ExpMaxScore_Diff_NS',(df['Score_NS']-df['ExpMaxScore_NS']),pl.Float32),
    )
    df = df.with_columns(
        pl.Series('ParScore_Diff_EW',-df['ParScore_Diff_NS'],pl.Int16), # used for open-closed room comparisons
        pl.Series('ExpMaxScore_Diff_EW',-df['ExpMaxScore_Diff_NS'],pl.Float32), # used for open-closed room comparisons
    )
    return df


def display_experiments(df):

    if 'Room' in df.columns and df['Room'].n_unique() == 2 and 'Open' in df['Room'].unique() and 'Closed' in df['Room'].unique():
        st.info("Following are WIP experiments showing comparative statistics for Open-Closed room competions. Comparisons include NS vs EW, tricks taken vs DD, par diffs, expected max value diffs.")

        for d in ['NS','EW']:
            g = df.group_by([d[0],d[1],'Room'])
            for k,v in g:
                st.caption(f"Summarize {k[2]} {d} ({k[0]}-{k[1]}) ParScore_Diff_{d}")
                sql_query = f"SUMMARIZE SELECT ParScore_Diff_{d}, DDTricks_Diff, ExpMaxScore_Diff_{d} FROM df WHERE Room='{k[2]}'" # DDTicks is directionally invariant
                ShowDataFrameTable(df, query=sql_query, key=f"display_experiments_{d+'_'.join(k)}_summarize")

        # # sum over Par_Diff_NS for all, bencam22, wbridge5

        # # todo: change f' to f" for all f strings
        # all, ns, ew = df[f'Par_Diff_{d}'].sum(),df[df['N'].eq('BENCAM22')]['Par_Diff_NS'].sum(),df[df['N'].eq('WBridge5')]['Par_Diff_NS'].sum()
        # st.write(f"Sum of Par_Diff_NS: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

        # # frequency where par was exceeded for all, bencam22, wbridge5
        # all, bencam22, wbridge5 = sum(df[f'Par_Diff_{d}'].gt(0)),sum(df['N'].eq('BENCAM22')&df['Par_Diff_NS'].gt(0)),sum(df['N'].eq('WBridge5')&df['Par_Diff_NS'].gt(0))
        # st.write(f"Frequency where exceeding Par: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

        # # describe() over DDTricks_Diff for all, bencam22, wbridge5
        # st.write('Describe Declarer, BENCAM22, DDTricks_Diff:')
        # st.write(df[df['DeclarerName'].eq('BENCAM22')]['DDTricks_Diff'].describe())
        # st.write('Describe Declarer, WBridge5, DDTricks_Diff:')
        # st.write(df[df['DeclarerName'].eq('WBridge5')]['DDTricks_Diff'].describe())

        # # sum over DDTricks_Diff for all, bencam22, wbridge5
        # all, bencam22, wbridge5 = df['DDTricks_Diff'].sum(),df[df['DeclarerName'].eq('BENCAM22')]['DDTricks_Diff'].sum(),df[df['DeclarerName'].eq('WBridge5')]['DDTricks_Diff'].sum()
        # st.write(f"Sum of DDTricks_Diff: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

        # # frequency where Tricks > DD for all, bencam22, wbridge5
        # all, bencam22, wbridge5 = sum(df['DDTricks_Diff'].notna() & df['DDTricks_Diff'].gt(0)),sum(df[df['DeclarerName'].eq('BENCAM22')]['DDTricks_Diff'].gt(0)),sum(df[df['DeclarerName'].eq('WBridge5')]['DDTricks_Diff'].gt(0))
        # st.write(f"Frequency where Tricks > DD: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

        # # frequency where Tricks < DD for all, bencam22, wbridge5
        # all, bencam22, wbridge5 = sum(df['DDTricks_Diff'].notna() & df['DDTricks_Diff'].lt(0)),sum(df[df['DeclarerName'].eq('BENCAM22')]['DDTricks_Diff'].lt(0)),sum(df[df['DeclarerName'].eq('WBridge5')]['DDTricks_Diff'].lt(0))
        # st.write(f"Frequency where Tricks < DD: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

        # # describe() over Par_Diff_NS for all, open, closed
        # st.write(df['Par_Diff_NS'].describe(),df[df['Room'].eq('Open')]['Par_Diff_NS'].describe(),df[df['Room'].eq('Closed')]['Par_Diff_NS'].describe())
        # # sum over Par_Diff_NS for all, bencam22, wbridge5
        # all, bencam22, wbridge5 = df['Par_Diff_NS'].sum(),df[df['Room'].eq('Open')]['Par_Diff_NS'].sum(),df[df['Room'].eq('Closed')]['Par_Diff_NS'].sum()
        # st.write(f"Sum of Par_Diff_NS: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")
        # all, open, closed = sum(df['Par_Diff_NS'].gt(0)),sum(df['Room'].eq('Open')&df['Par_Diff_NS'].gt(0)),sum(df['Room'].eq('Closed')&df['Par_Diff_NS'].gt(0))
        # st.write(f"Frequency where exceeding Par: All:{all} Open:{open} Closed:{closed} Open-Closed:{open-closed}")

        # # describe() over ExpMaxScore_Diff_NS for all, open, closed
        # st.write(df['ExpMaxScore_Diff_NS'].describe(),df[df['Room'].eq('Open')]['ExpMaxScore_Diff_NS'].describe(),df[df['Room'].eq('Closed')]['ExpMaxScore_Diff_NS'].describe())
        # # sum over ExpMaxScore_Diff_NS for all, bencam22, wbridge5
        # all, bencam22, wbridge5 = df['ExpMaxScore_Diff_NS'].sum(),df[df['Room'].eq('Open')]['ExpMaxScore_Diff_NS'].sum(),df[df['Room'].eq('Closed')]['ExpMaxScore_Diff_NS'].sum()
        # st.write(f"Sum of ExpMaxScore_Diff_NS: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")
        # all, open, closed = sum(df['ExpMaxScore_Diff_NS'].gt(0)),sum(df['Room'].eq('Open')&df['ExpMaxScore_Diff_NS'].gt(0)),sum(df['Room'].eq('Closed')&df['ExpMaxScore_Diff_NS'].gt(0))
        # st.write(f"Frequency where exceeding ExpMaxScore_Diff_NS: All:{all} Open:{open} Closed:{closed} Open-Closed:{open-closed}")


def ShowDataFrameTable(df, key, query='SELECT * FROM df'):
    if st.session_state.show_sql_query:
        st.caption(f"SQL Query: {query}")
    try:
        df = duckdb.sql(query).df() # returns a pandas dataframe
    except Exception as e:
        st.error(f"Invalid SQL Query: {query} {e}")
        return True
    try:
        streamlitlib.ShowDataFrameTable(df, key)
    except Exception as e:
        st.error(f"Invalid SQL Query on Dataframe: {query} {e}")
        return True
    return False



def app_info():
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in Streamlit. Data engine is Pandas. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/Calculate_PBN_Results")
    st.caption(
        f"App:{app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__} Query Params:{st.query_params.to_dict()}")


from fsspec.utils import infer_storage_options

def get_url_protocol(path):
    # Use fsspec to infer the storage type
    options = infer_storage_options(path)
    # Check if the protocol is 'file', which indicates a local file
    return options['protocol']


def chat_input_on_submit():
    prompt = st.session_state.main_prompt_chat_input
    ShowDataFrameTable(st.session_state.df, query=prompt, key='user_query_main_doit')


def sample_count_on_change():
    st.session_state.single_dummy_sample_count = st.session_state.create_sidebar_single_dummy_sample_count
    if 'df' in st.session_state:
        LoadPage()


def sql_query_on_change():
    st.session_state.show_sql_query = st.session_state.create_sidebar_show_sql_query_checkbox
    if 'df' in st.session_state:
        LoadPage()
    #pass # do nothing


def LoadPage():

    #with st.session_state.chat_container:

    url = st.session_state.create_sidebar_text_input_url
    st.caption(f"Selected PBN file: {url}") # using protocol:{get_url_protocol(url)}")

    if url is None or url == '' or (get_url_protocol(url) == 'file' and ('/' in url and '\\' in url and '&' in url)):
        return

    path_url = pathlib.Path(url)
    boards = None
    df = None
    everything_df = None
    if url.endswith('_boards.pkl'):
        if not path_url.exists():
            st.warning(f"{url} does not exist.")
            return
        with st.spinner(f"Loading {url} ..."):
            with open(path_url, 'rb') as f:
                boards = pickle.load(f)
        url = url.replace('_boards.pkl','')
        path_url = pathlib.Path(url)
    elif url.endswith('_df.pkl'):
        if not path_url.exists():
            st.warning(f"{url} does not exist.")
            return
        with st.spinner(f"Loading {url} ..."):
            with open(path_url, 'rb') as f:
                df = pickle.load(f)
        url = url.replace('_df.pkl','')
        path_url = pathlib.Path(url)
    elif url.endswith('_everythingdf.parquet'):
        if not path_url.exists():
            st.warning(f"{url} does not exist.")
            return
        with st.spinner(f"Loading {url} ..."):
            everything_df = pl.read_parquet(path_url)
        url = url.replace('_everythingdf.parquet','')
        path_url = pathlib.Path(url)
    else:
        with st.spinner(f"Loading {url} ..."):
            of = fsspec.open(url, mode='r', encoding='utf-8')
            with of as f:
                pbn_data = f.read()

        if boards is None and df is None:
            with st.spinner("Parsing PBN file ..."):
                boards = pbn.loads(pbn_data)
                if len(boards) == 0:
                    st.warning(f"{url} has no boards.")
                    return
                if len(boards) > recommended_board_max:
                    st.warning(f"{url} has {len(boards)} boards. More than {recommended_board_max} boards may result in instability.")
        if save_files:
            boards_url = pathlib.Path(path_url.stem+'_boards').with_suffix('.pkl')
            boards_path = pathlib.Path(boards_url)
            with st.spinner(f"Saving {boards_url} file ..."):
                with open(boards_path, 'wb') as f:
                    pickle.dump(boards, f)
                st.caption(f"Saved {boards_url}. File length is {boards_path.stat().st_size} bytes.")

    if everything_df is None:
        if df is None:
            with st.spinner("Creating PBN Dataframe ..."):
                #st.write(f"Number of boards: {len(boards)}")
                #st.write(vars(boards[0]))
                df = create_df_from_pbn(boards)
            with st.spinner("Cleaning and Augmenting PBN Dataframe ..."):
                # Other dataframe components don't do implicit str conversion like pl.DataFrame. Must manually convert object columns to strings.
                df = df.with_columns(
                    pl.Series('Deal',map(str,df['deal']),pl.String),
                    pl.Series('Dealer',map(lambda x: 'NESW'[x],df['_dealer']),pl.String),
                    pl.Series('Vul',map(lambda x: ['None','NS','EW','Both'][x],df['_vul']),pl.String),
                    pl.Series('Auction',map(lambda x: ', '.join(map(str,x[:3]))+' ...',df['auction']),pl.String),
                    pl.Series('Play',map(lambda x: ', '.join(map(str,x[:3]))+' ...',df['play']),pl.String),
                    pl.Series('Contract',map(str,df['_contract']),pl.String),
                )
                df = df.rename({'board_num':'Board','claimed':'Claimed'})
                # todo: have to exclude auction: ArrowInvalid: Could not convert PenaltyBid(penalty=<Penalty.passed: 1>, alertable=False, announcement=None) with type PenaltyBid: did not recognize Python value type when inferring an Arrow data type
                # if save_files:
                #     # save df as pickle because it contains object columns. later, they're dropped when creating pbn_df.
                #     df_url = pathlib.Path(path_url.stem+'_df').with_suffix('.pkl')
                #     df_path = pathlib.Path(df_url)
                #     with st.spinner(f"Saving {df_url} file ..."):
                #         with open(df_path, 'wb') as f:
                #             exclude_columns = ['deal','_dealer','_vul','auction','play','_contract'] # todo: fix this. shouldn't have to exclude. object issue.
                #             pickle.dump(df.select(pl.exclude(exclude_columns)), f)
                #     st.caption(f"Saved {df_url}. File length is {df_path.stat().st_size} bytes.")
        exclude_columns = ['deal','_dealer','_vul','auction','play','_contract']
        column_order = ['Date','Scoring','Board','Room','Deal','North','East','South','West','Dealer','Vul','Auction','Contract','Play','Score','Claimed','Event','Site','BCFlags']
        column_order = [c for c in column_order if c in df.columns]
        # add any not-well-known columns but prepend with underscore to avoid conflicts
        for c in df.columns:
            if c not in column_order:
                if c not in exclude_columns:
                    custom_c = 'Custom_'+c
                    df = df.rename({c:custom_c})
                    column_order.append(custom_c)
        pbn_df = df.select(pl.col(column_order))
    
        st.caption("PBN Dataframe")
        ShowDataFrameTable(pbn_df, key='LoadPage_pbn_df')

        DDTricks_progress = st.progress(0,"Calculating Double Dummy Tricks")
        DDTricks_df, par_df, dd_score_df = calculate_ddtricks_par_scores(df, progress=DDTricks_progress)
        st.caption("Double Dummy Tricks Dataframe")
        ShowDataFrameTable(DDTricks_df, key='LoadPage_DDTricks_df')
        st.caption("Par Scores Dataframe")
        ShowDataFrameTable(par_df, key='LoadPage_par_df')
        st.caption("Double Dummy Scores Dataframe")
        ShowDataFrameTable(dd_score_df, key='LoadPage_dd_score_df')

        everything_df = pl.concat([pbn_df,DDTricks_df,par_df,dd_score_df],how='horizontal')
        if st.session_state.single_dummy_sample_count:
            st.session_state.single_dummy_sample_count = single_dummy_sample_count_default
            sd_prob_progress = st.progress(0,f"Calculating Single Dummy Probabilities from {st.session_state.single_dummy_sample_count} Samples")
            sd_cache_d, sd_probs_df = calculate_sd_probs(df, st.session_state.single_dummy_sample_count, progress=sd_prob_progress)
            st.caption(f"Single Dummy Probabilities Dataframe Using {st.session_state.single_dummy_sample_count} Samples")
            ShowDataFrameTable(sd_probs_df, key='LoadPage_sd_probs_df')

            scores_d, scores_df = calculate_scores()
            st.caption("Scores Dataframe (not vul, vul)")
            ShowDataFrameTable(scores_df, key='LoadPage_scores_df')

            sd_expected_values_df = calculate_sd_expected_values(df, sd_cache_d, scores_d)
            st.caption("Single Dummy Expected Values Dataframe")
            ShowDataFrameTable(sd_expected_values_df, key='LoadPage_sd_expected_values_df')

            sd_best_contract_df = calculate_best_contracts(sd_expected_values_df)
            st.caption("Single Dummy Best Contracts Dataframe")
            ShowDataFrameTable(sd_best_contract_df, key='LoadPage_sd_best_contract_df')
            everything_df = pl.concat([everything_df,sd_probs_df,scores_df,sd_expected_values_df,sd_best_contract_df],how='horizontal')
            everything_df = create_augmented_df(everything_df)
        if save_files:
            everythingdf_url = pathlib.Path(path_url.stem+'_everythingdf').with_suffix('.parquet')
            everythingdf_path = pathlib.Path(everythingdf_url)
            with st.spinner(f"Saving {everythingdf_url} file ..."):
                everything_df.write_parquet(everythingdf_path)
            st.caption(f"Saved {everythingdf_url}. File length is {everythingdf_path.stat().st_size} bytes.")

    st.caption("Everything Dataframe")
    ShowDataFrameTable(everything_df, key='LoadPage_everything_df')

    display_experiments(everything_df)

    st.session_state.df = everything_df

    st.caption("All dataframes have been calculated and displayed.")
    st.caption("You may now enter SQL queries in the chat box below. Use *FROM df* to query the everything dataframe.")


def create_sidebar():
    st.sidebar.caption('Build:'+app_datetime)

    # example valid urls
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
    #default_url = 'file://c:/sw/bridge/ML-Contract-Bridge/src/Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://c:\sw/bridge\ML-Contract-Bridge\src\Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = 'GIB-Thorvald-8638-2024-08-23.pbn'
    st.sidebar.text_input('Enter PBN URL:', default_url, key='create_sidebar_text_input_url', help='Enter a URL or pathless local file name.') # , on_change=LoadPage
    st.sidebar.button('Go', on_click=LoadPage, key='create_sidebar_go_button', help='Load PBN data from URL.')

    st.sidebar.number_input('Single Dummy Sample Count',value=single_dummy_sample_count_default,key='create_sidebar_single_dummy_sample_count',on_change=sample_count_on_change,min_value=1,max_value=1000,step=1,help='Number of random deals to generate for calculating single dummy probabilities. Larger number (10 to 30) is more accurate but slower. Use 1 to 5 for fast, less accurate results.')

    # SELECT Board, Vul, ParContract, ParScore_NS, Custom_ParContract FROM df
    st.sidebar.checkbox('Show SQL Query',value=show_sql_query_default,key='create_sidebar_show_sql_query_checkbox',on_change=sql_query_on_change,help='Show SQL used to query dataframes.')


if __name__ == '__main__':

    # Configurations
    app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    #pbn_filename_default = 'DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'  # local filename
    single_dummy_sample_count_default = 2  # number of random deals to generate for calculating single dummy probabilities. Use smaller number for testing.
    st.session_state.single_dummy_sample_count = single_dummy_sample_count_default
    show_sql_query_default = True
    st.session_state.show_sql_query = show_sql_query_default
    save_files = True
    recommended_board_max = 10000

    create_sidebar()

    if 'df' not in st.session_state:
        st.title("Calculate PBN Deal Statistics")
        app_info()
        st.info("*Start by clicking the Go button on the left sidebar.*")
    else:
        st.chat_input('Enter a SQL query e.g. SELECT * FROM df', key='main_prompt_chat_input', on_submit=chat_input_on_submit)

