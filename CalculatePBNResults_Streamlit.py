
# streamlit program to display deal statistics from a PBN file.
# Invoke from system prompt using: streamlit run CalculatePBNResults_Streamlit.py

import streamlit as st
import pathlib
import fsspec
import pandas as pd # only used for __version__ for now. might need for plotting later as pandas plotting support is better than polars.
import polars as pl
from collections import defaultdict
from datetime import datetime, timezone
import sys

import endplay # for __version__
from endplay.parsers import pbn
from endplay.types import Deal, Contract, Denom, Player, Penalty
from endplay.dds import par, calc_all_tables
from endplay.dealer import generate_deals

sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
import streamlitlib


def create_df_from_pbn(boards):
    # Create dataframe from boards
    df = pl.DataFrame([vars(b) for b in boards])
    for col in df.columns:
        #st.write(f"dtype:{df[col].dtype} Column:{col} Height:{df.height} Type:{df[col].dtype}")
        if df.height:
            if isinstance(df[col].dtype, pl.Struct):
                if isinstance(df[col][0], dict):
                    #st.write(f"Struct Column:{col} Dict:{df[col][0]}")
                    df = df.unnest(col) # unnest contents of dict into columns for each key:value pair
    return df


def calculate_dd_tricks_pars_scores(df, progress=None):

    NSEW_direction_order = [0, 2, 1, 3]
    SHDCN_suit_order = [3, 2, 1, 0, 4]
    # Calculate double dummy and par
    deals = df['deal'] # using deal object here
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

    # Create dataframe of par scores (double dummy)
    pars = [par(tt, b, 0) for tt, b in zip(tables, df['board_num'])]  # middle arg is board (if int) otherwise enum vul.
    par_scores_ns = [parlist.score for parlist in pars]
    par_scores_ew = [-score for score in par_scores_ns]
    par_contracts = [', '.join([str(contract.level) + 'SHDCN'[int(contract.denom)] + contract.declarer.abbr + contract.penalty.abbr + ('' if contract.result == 0 else '+'+str(contract.result) if contract.result > 0 else str(contract.result)) for contract in parlist]) for parlist in pars]
    par_df = pl.DataFrame({'Par_Score_NS': par_scores_ns, 'Par_Score_EW': par_scores_ew, 'Par_Contract': par_contracts})

    # Create dataframe of double dummy tricks per direction and suit
    dd_tricks_rows = [[sd[suit][direction] for direction in NSEW_direction_order for suit in SHDCN_suit_order] for sd in t_t]
    dd_tricks_df = pl.DataFrame(dd_tricks_rows, schema=['_'.join(['DD_Tricks', d, s]) for d in 'NSEW' for s in 'CDHSN'],orient='row')

    def Tricks_To_Score(sd):
        return [Contract(level=level, denom=suit, declarer=direction, penalty=Penalty.passed if sd[suit][direction] - 6 - level >= 0 else Penalty.doubled, result=sd[suit][direction] - 6 - level).score(0) for direction in NSEW_direction_order for suit in SHDCN_suit_order for level in range(1, 8)]

    dd_score_rows = [Tricks_To_Score(sd) for sd in t_t]
    dd_score_df = pl.DataFrame(dd_score_rows, schema=['_'.join(['DD_Score', str(l) + s, d]) for d in 'NSEW' for s in 'CDHSN' for l in range(1, 8)],orient='row')

    return dd_tricks_df, par_df, dd_score_df


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
        for d in 'NSEW':
            for s in 'SHDCN':
                # todo: convert this line from pandas to polars
                ns_ew_rows[(ns_ew,d,s)] = dd_df[d+s].to_pandas().value_counts(normalize=True).reindex(range(14), fill_value=0).tolist() # ['Fixed_Direction','Direction_Declarer','Suit']+['SD_Prob_Take_'+str(n) for n in range(14)]
    
    return ns_ew_rows


# def append_single_dummy_results(pbns,sd_cache_d,produce=100):
#     for pbn in pbns:
#         if pbn not in sd_cache_d:
#             sd_cache_d[pbn] = calculate_single_dummy_probabilities(pbn, produce) # all combinations of declarer pair direction, declarer direciton, suit, tricks taken
#     return sd_cache_d


# takes 1000 seconds for 100 sd calcs, or 10 sd calcs per second.
def calculate_sd_probs(df, sd_productions=100, progress=None):
    sd_cache_d = {}
    deals = list(map(str,df['deal'])) # using str of deal here
    for i,deal in enumerate(deals):
        if progress:
            percent_complete = int(i*100/len(deals))
            progress.progress(percent_complete,f"{percent_complete}%: {i} of {len(deals)} single dummies calculated. deal:{deal}")
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
        for (pair_direction,declarer_direction,suit),tricks in v.items():
            for i,t in enumerate(tricks):
                sd_probs_d['_'.join(['Probs',pair_direction,declarer_direction,suit,str(i)])].append(t)
    # st.write(sd_probs_d)
    sd_probs_df = pl.DataFrame(sd_probs_d)
    return sd_cache_d, sd_probs_df


# calculate dict of contract result scores
def calculate_sd_scores():
    SHDCN_suit_order = [3, 2, 1, 0, 4]

    sd_scores_d = {}
    for suit in SHDCN_suit_order:
        for level in range(1,8): # contract level
            for tricks in range(14):
                result = tricks-6-level
                sd_scores_d[(level,'SHDCN'[suit],tricks,False)] = Contract(level=level,denom=suit,declarer=0,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(False)
                sd_scores_d[(level,'SHDCN'[suit],tricks,True)] = Contract(level=level,denom=suit,declarer=0,penalty=Penalty.passed if result>=0 else Penalty.doubled,result=result).score(True)

    # create score dataframe from dict
    scores_d = defaultdict(list)
    for suit in 'SHDCN':
        for level in range(1,8):
            for i in range(14):
                scores_d['_'.join(['Score',str(level)+suit])].append([sd_scores_d[(level,suit,i,False)],sd_scores_d[(level,suit,i,True)]])
    # st.write(scores_d)
    sd_scores_df = pl.DataFrame(scores_d)
    #sd_scores_df.index.name = 'Taken'
    return scores_d, sd_scores_df


def calculate_sd_expected_values(df,sd_cache_d,scores_d):
    # create dict of expected values (probability * score)
    exp_d = defaultdict(list)
    deal_vul = zip(map(str,df['deal']),df['_vul']) # using str of deal here
    for deal,vul in deal_vul:
        #st.write(deal,vul)
        for (pair_direction,declarer_direction,suit),probs in sd_cache_d[deal].items():
            is_vul = vul == 1 or (declarer_direction in 'NS' and vul == 2) or (declarer_direction in 'EW' and vul == 3)
            #st.write(pair_direction,declarer_direction,suit,probs,is_vul)
            for level in range(1,8):
                #st.write(scores_d['_'.join(['Score',str(level)+suit])][is_vul])
                exp_d['_'.join(['Exp',pair_direction,declarer_direction,suit,str(level)])].append(sum([prob*score[is_vul] for prob,score in zip(probs,scores_d['_'.join(['Score',str(level)+suit])])]))
            #st.write(exp_d)
    #st.write(exp_d)
    sd_exp_df = pl.DataFrame(exp_d)
    return sd_exp_df


# create columns containing the 1) the name of the column having the max expected value. 2) max expected value 3) contract having the max expected value.
def create_best_contracts(r):
    exp_tuples = tuple([(v,k) for k,v in r.items()])
    ex_tuples_sorted = sorted(exp_tuples,reverse=True)
    best_contract_tuple = ex_tuples_sorted[0]
    best_contract_split = best_contract_tuple[1].split('_') # split column name into parts
    best_contract = best_contract_split[4]+best_contract_split[3]+best_contract_split[2]
    return [best_contract_tuple[1],best_contract_tuple[0],best_contract_tuple[0] if best_contract_tuple[1][-5] in ['N','S'] else -best_contract_tuple[0],best_contract]


def calculate_best_contracts(sd_exp_df):
    # todo: convert from pandas to polars.
    sd_best_contract_l = sd_exp_df.to_pandas().apply(create_best_contracts,axis='columns')
    sd_best_contract_df = pl.DataFrame(sd_best_contract_l.tolist(),schema=['Exp_Max_Col','Exp_Max','Exp_Max_NS','Best_Contract'],orient='row')
    return sd_best_contract_df


def convert_contract_to_contract(df):
    return df['_contract'].str.to_uppercase().str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C').str.replace('NT','N')


# None is used instead of pl.Null because pl.Null becomes 'Null' string in pl.String columns. Not sure what's going on but the solution is to use None.
def convert_contract_to_declarer(df):
    return [None if c == 'PASS' else c[2] for c in df['Contract']] # extract declarer from contract


def convert_declarer_to_declarer_name(df):
    return [None if d is None else df[d][i] for i,d in enumerate(df['Declarer'])] # extract declarer name using declarer direction as the lookup key


def convert_contract_to_result(df):
    return [None if c == 'PASS' else 0 if c[-1] in ['=','0'] else int(c[-1]) if c[-2] == '+' else -int(c[-1]) for c in df['Contract']] # create result from contract


def convert_contract_to_tricks(df):
    return [None if c == 'PASS' else int(c[0])+6+r for c,r in zip(df['Contract'],df['Result'])] # create tricks from contract and result


def convert_contract_to_dd_tricks(df):
    return [None if c == 'PASS' else df['_'.join(['DD_Tricks',d,c[1]])][i] for i,(c,d) in enumerate(zip(df['Contract'],df['Declarer']))] # extract double dummy tricks using contract and declarer as the lookup keys


def convert_score_to_score(df):
    score_split = r['Score'].split()
    assert len(score_split) == 2, f"score_split:{score_split}"
    assert score_split[0] in ['NS','EW'], f"score_split:{score_split[0]}"
    assert score_split[1][0] == '-' or str.isdigit(score_split[1][0]), f"score_split:{score_split[1]}"
    score_split_direction = score_split[0]
    score_split_value = score_split[1]
    score_value = -int(score_split_value) if score_split_value[0] == '-' else int(score_split_value)
    return score_value if score_split_direction == 'NS' else -score_value


def create_augmented_df(merged_df):
    augmented_df = merged_df.clone()
    augmented_df = augmented_df.rename({'North':'N','East':'E','South':'S','West':'W'}) # todo: is this really better?

    augmented_df = augmented_df.with_columns(
        pl.Series('Contract',convert_contract_to_contract(augmented_df),pl.String,strict=False), # can have nulls or Strings
    )
    augmented_df = augmented_df.with_columns(
        pl.Series('Declarer',convert_contract_to_declarer(augmented_df),pl.String,strict=False), # can have nulls or Strings
    )
    augmented_df = augmented_df.with_columns(
        pl.Series('Declarer_Name',convert_declarer_to_declarer_name(augmented_df),pl.String,strict=False), # can have nulls or Strings
        pl.Series('Result',convert_contract_to_result(augmented_df),pl.Int8,strict=False), # can have nulls or Int8
    )
    augmented_df = augmented_df.with_columns(
        pl.Series('Tricks',convert_contract_to_tricks(augmented_df),pl.UInt8,strict=False), # can have nulls or UInt8
        pl.Series('DD_Tricks',convert_contract_to_dd_tricks(augmented_df),pl.UInt8,strict=False), # can have nulls or UInt8
        #pl.Series('Score_NS',convert_score_to_score(augmented_df),pl.Int16),
    )
    augmented_df = augmented_df.with_columns(
        #pl.Series('Score_Par_Diff_NS',(augmented_df['Score_NS']-augmented_df['Par_Score_NS']),pl.Int16,strict=False), # can have nulls or Int16
        # needs to have .cast(pl.Int8) because left and right are both UInt8 which goofs up the subtraction.
        pl.Series('Tricks_DDTricks_Diff',(augmented_df['Tricks'].cast(pl.Int8)-augmented_df['DD_Tricks'].cast(pl.Int8)),pl.Int8,strict=False), # can have nulls or Int8
        #pl.Series('Score_ExpMax_Diff_NS',(augmented_df['Score_NS']-augmented_df['Exp_Max_NS']),pl.Float32,strict=False), # can have nulls or Float32
    )
    return augmented_df


def display_experiments(augmented_df):

    st.header("Following cells contain WIP experiments with comparative statistics; BENCAM22 vs WBridge5, Open vs Closed rooms, Tricks vs DD, par diffs, expected max diffs.")

    # describe() over Par_Diff_NS for all, bencam22, wbridge5
    st.write('Describe North, BENCAM22, Par_Diff_NS:')
    st.write(augmented_df[augmented_df['N'].eq('BENCAM22')]['Par_Diff_NS'].describe())
    st.write('Describe North, WBridge5, Par_Diff_NS:')
    st.write(augmented_df[augmented_df['N'].eq('WBridge5')]['Par_Diff_NS'].describe())

    # sum over Par_Diff_NS for all, bencam22, wbridge5
    all, bencam22, wbridge5 = augmented_df['Par_Diff_NS'].sum(),augmented_df[augmented_df['N'].eq('BENCAM22')]['Par_Diff_NS'].sum(),augmented_df[augmented_df['N'].eq('WBridge5')]['Par_Diff_NS'].sum()
    st.write(f"Sum of Par_Diff_NS: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

    # frequency where par was exceeded for all, bencam22, wbridge5
    all, bencam22, wbridge5 = sum(augmented_df['Par_Diff_NS'].gt(0)),sum(augmented_df['N'].eq('BENCAM22')&augmented_df['Par_Diff_NS'].gt(0)),sum(augmented_df['N'].eq('WBridge5')&augmented_df['Par_Diff_NS'].gt(0))
    st.write(f"Frequency where exceeding Par: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

    # describe() over DD_Tricks_Diff for all, bencam22, wbridge5
    st.write('Describe Declarer, BENCAM22, DD_Tricks_Diff:')
    st.write(augmented_df[augmented_df['Declarer_Name'].eq('BENCAM22')]['DD_Tricks_Diff'].describe())
    st.write('Describe Declarer, WBridge5, DD_Tricks_Diff:')
    st.write(augmented_df[augmented_df['Declarer_Name'].eq('WBridge5')]['DD_Tricks_Diff'].describe())

    # sum over DD_Tricks_Diff for all, bencam22, wbridge5
    all, bencam22, wbridge5 = augmented_df['DD_Tricks_Diff'].sum(),augmented_df[augmented_df['Declarer_Name'].eq('BENCAM22')]['DD_Tricks_Diff'].sum(),augmented_df[augmented_df['Declarer_Name'].eq('WBridge5')]['DD_Tricks_Diff'].sum()
    st.write(f"Sum of DD_Tricks_Diff: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

    # frequency where Tricks > DD for all, bencam22, wbridge5
    all, bencam22, wbridge5 = sum(augmented_df['DD_Tricks_Diff'].notna() & augmented_df['DD_Tricks_Diff'].gt(0)),sum(augmented_df[augmented_df['Declarer_Name'].eq('BENCAM22')]['DD_Tricks_Diff'].gt(0)),sum(augmented_df[augmented_df['Declarer_Name'].eq('WBridge5')]['DD_Tricks_Diff'].gt(0))
    st.write(f"Frequency where Tricks > DD: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

    # frequency where Tricks < DD for all, bencam22, wbridge5
    all, bencam22, wbridge5 = sum(augmented_df['DD_Tricks_Diff'].notna() & augmented_df['DD_Tricks_Diff'].lt(0)),sum(augmented_df[augmented_df['Declarer_Name'].eq('BENCAM22')]['DD_Tricks_Diff'].lt(0)),sum(augmented_df[augmented_df['Declarer_Name'].eq('WBridge5')]['DD_Tricks_Diff'].lt(0))
    st.write(f"Frequency where Tricks < DD: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")

    # describe() over Par_Diff_NS for all, open, closed
    st.write(augmented_df['Par_Diff_NS'].describe(),augmented_df[augmented_df['Room'].eq('Open')]['Par_Diff_NS'].describe(),augmented_df[augmented_df['Room'].eq('Closed')]['Par_Diff_NS'].describe())
    # sum over Par_Diff_NS for all, bencam22, wbridge5
    all, bencam22, wbridge5 = augmented_df['Par_Diff_NS'].sum(),augmented_df[augmented_df['Room'].eq('Open')]['Par_Diff_NS'].sum(),augmented_df[augmented_df['Room'].eq('Closed')]['Par_Diff_NS'].sum()
    st.write(f"Sum of Par_Diff_NS: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")
    all, open, closed = sum(augmented_df['Par_Diff_NS'].gt(0)),sum(augmented_df['Room'].eq('Open')&augmented_df['Par_Diff_NS'].gt(0)),sum(augmented_df['Room'].eq('Closed')&augmented_df['Par_Diff_NS'].gt(0))
    st.write(f"Frequency where exceeding Par: All:{all} Open:{open} Closed:{closed} Open-Closed:{open-closed}")

    # describe() over Exp_Max_Diff_NS for all, open, closed
    st.write(augmented_df['Exp_Max_Diff_NS'].describe(),augmented_df[augmented_df['Room'].eq('Open')]['Exp_Max_Diff_NS'].describe(),augmented_df[augmented_df['Room'].eq('Closed')]['Exp_Max_Diff_NS'].describe())
    # sum over Exp_Max_Diff_NS for all, bencam22, wbridge5
    all, bencam22, wbridge5 = augmented_df['Exp_Max_Diff_NS'].sum(),augmented_df[augmented_df['Room'].eq('Open')]['Exp_Max_Diff_NS'].sum(),augmented_df[augmented_df['Room'].eq('Closed')]['Exp_Max_Diff_NS'].sum()
    st.write(f"Sum of Exp_Max_Diff_NS: All:{all} BENCAM22:{bencam22} WBridge5:{wbridge5} BENCAM22-WBridge5:{bencam22-wbridge5}")
    all, open, closed = sum(augmented_df['Exp_Max_Diff_NS'].gt(0)),sum(augmented_df['Room'].eq('Open')&augmented_df['Exp_Max_Diff_NS'].gt(0)),sum(augmented_df['Room'].eq('Closed')&augmented_df['Exp_Max_Diff_NS'].gt(0))
    st.write(f"Frequency where exceeding Exp_Max_Diff_NS: All:{all} Open:{open} Closed:{closed} Open-Closed:{open-closed}")

    
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


def LoadPage(url):

    with st.container():
        #url = st.session_state.text_input_url
        st.write(f"URL:{url} protocol:{get_url_protocol(url)}")

        if url is None or url == '' or (get_url_protocol(url) == 'file' and ('/' in url and '\\' in url and '&' in url)):
            return

        # example valid urls
        #url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
        #url = 'file://c:/sw/bridge/ML-Contract-Bridge/src/Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
        #url = r'file://c:\sw/bridge\ML-Contract-Bridge\src\Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
        url = r'file://DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
        of = fsspec.open(url, mode='r', encoding='utf-8')
        with of as f:
            pbn_data = f.read()

        boards = pbn.loads(pbn_data)

        #st.write(f"Number of boards: {len(boards)}")
        #st.write(vars(boards[0]))

        df = create_df_from_pbn(boards)
        st.caption("Information Dataframe")
        # Other dataframe components don't do implicit str conversion like pl.DataFrame. Must manually convert object columns to strings.
        str_df = df.clone()
        str_df = str_df.with_columns(
            pl.Series('deal',map(str,str_df['deal']),pl.String),
            pl.Series('auction',map(lambda x: ', '.join(map(str,x[:3]))+' ...',str_df['auction']),pl.String),
            pl.Series('play',map(lambda x: ', '.join(map(str,x[:3]))+' ...',str_df['play']),pl.String),
            pl.Series('_contract',map(str,str_df['_contract']),pl.String),
        )
        streamlitlib.ShowDataFrameTable(str_df.to_pandas(), key='info_df')

        dd_tricks_progress = st.progress(0,"Calculating Double Dummy Tricks")
        dd_tricks_df, par_df, dd_score_df = calculate_dd_tricks_pars_scores(df, progress=dd_tricks_progress)
        st.caption("Double Dummy Tricks Dataframe")
        streamlitlib.ShowDataFrameTable(dd_tricks_df.to_pandas(), key='dd_tricks_df')
        st.caption("Par Scores Dataframe")
        streamlitlib.ShowDataFrameTable(par_df.to_pandas(), key='par_df')
        st.caption("Double Dummy Scores Dataframe")
        streamlitlib.ShowDataFrameTable(dd_score_df.to_pandas(), key='dd_score_df')

        sd_prob_progress = st.progress(0,"Calculating Single Dummy Probabilities")
        sd_cache_d, sd_probs_df = calculate_sd_probs(df, sd_productions_default, progress=sd_prob_progress)
        st.caption("Single Dummy Probabilities Dataframe")
        streamlitlib.ShowDataFrameTable(sd_probs_df.to_pandas(), key='sd_probs_df')

        scores_d, sd_scores_df = calculate_sd_scores()
        st.caption("Single Dummy Scores Dataframe")
        streamlitlib.ShowDataFrameTable(sd_scores_df.to_pandas(), key='sd_scores_df')

        sd_expected_values_df = calculate_sd_expected_values(df, sd_cache_d, scores_d)
        st.caption("Expected Values Dataframe")
        streamlitlib.ShowDataFrameTable(sd_expected_values_df.to_pandas(), key='sd_expected_values_df')

        sd_best_contract_df = calculate_best_contracts(sd_expected_values_df)
        st.caption("Best Contracts Dataframe")
        streamlitlib.ShowDataFrameTable(sd_best_contract_df.to_pandas(), key='sd_best_contract_df')

        merged_df = pl.concat([str_df,dd_tricks_df,par_df,dd_score_df,sd_probs_df,sd_scores_df,sd_expected_values_df,sd_best_contract_df],how='horizontal')
        #st.caption("Merged Dataframe")
        #streamlitlib.ShowDataFrameTable(merged_df.to_pandas(), key='merged_df')
        augmented_df = create_augmented_df(merged_df)
        st.caption("Everything Dataframe")
        streamlitlib.ShowDataFrameTable(augmented_df.to_pandas(), key='augmented_df')

        display_experiments(augmented_df)


def create_sidebar():
    st.sidebar.caption('Build:'+app_datetime)

    default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
    url = st.sidebar.text_input('Enter PBN URL:', default_url, key='text_input_url') # on_change=LoadPage)

    if url is not None and url != '':
        LoadPage(url)


if __name__ == '__main__':

    # Configurations
    app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    pbn_filename_default = 'DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'  # local filename
    sd_productions_default = 2  # number of random deals to generate for calculating single dummy probabilities. Use smaller number for testing.

    with st.container():

        st.title("Calculate PBN Deal Statistics")
        app_info()

        create_sidebar()

