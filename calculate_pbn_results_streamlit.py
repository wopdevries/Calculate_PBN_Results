"""
PBN Results Calculator Streamlit Application
"""

# streamlit program to display Bridge game deal statistics from a PBN file.
# Invoke from system prompt using: streamlit run CalculatePBNResults_Streamlit.py


import streamlit as st
import streamlit_chat
from streamlit_extras.bottom_container import bottom
from stqdm import stqdm
from st_aggrid import AgGrid

import pathlib
import fsspec
import polars as pl
import duckdb
import json
import pickle
from collections import defaultdict
from datetime import datetime, timezone
import sys
import platform
from dotenv import load_dotenv

# Only declared to display version information
#import fastai
import numpy as np
import pandas as pd
#import safetensors
#import sklearn
#import torch

import endplay # for __version__
from endplay.parsers import pbn, lin, json
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import par, calc_all_tables
from endplay.dealer import generate_deals

sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global # Requires "./mlBridgeLib" be in extraPaths in .vscode/settings.json

import streamlitlib
from mlBridgeLib.mlBridgePostmortemLib import PostmortemBase
import mlBridgeEndplayLib
from mlBridgeLib.mlBridgeAugmentLib import (
    AllAugmentations,
)#import mlBridgeBiddingLib



# def create_augmented_df(df):
#     #df = df.clone()
#     df = df.rename({'North':'N','East':'E','South':'S','West':'W'}) # todo: is this really better?

#     df = df.with_columns(
#         pl.Series('Contract',convert_contract_to_contract(df),pl.String,strict=False), # can have nulls or Strings
#     )
#     df = df.with_columns(
#         pl.Series('Declarer',convert_contract_to_declarer(df),pl.String,strict=False), # can have nulls or Strings
#     )
#     df = df.with_columns(
#         pl.Series('DeclarerName',convert_declarer_to_DeclarerName(df),pl.String,strict=False), # can have nulls or Strings
#         pl.Series('Result',convert_contract_to_result(df),pl.Int8,strict=False), # can have nulls or Int8
#     )
#     df = df.with_columns(
#         pl.Series('Tricks',convert_contract_to_tricks(df),pl.UInt8,strict=False), # can have nulls or UInt8
#         pl.Series('DDTricks',convert_contract_to_DDTricks(df),pl.UInt8,strict=False), # can have nulls or UInt8
#         pl.Series('Score_NS',convert_score_to_score(df),pl.Int16),
#     )
#     df = df.with_columns(
#         pl.Series('ParScore_Diff_NS',(df['Score_NS']-df['ParScore_NS']),pl.Int16),
#         # needs to have .cast(pl.Int8) because left and right are both UInt8 which goofs up the subtraction.
#         pl.Series('DDTricks_Diff',(df['Tricks'].cast(pl.Int8)-df['DDTricks'].cast(pl.Int8)),pl.Int8,strict=False), # can have nulls or Int8
#         pl.Series('ExpMaxScore_Diff_NS',(df['Score_NS']-df['ExpMaxScore_NS']),pl.Float32),
#     )
#     df = df.with_columns(
#         pl.Series('ParScore_Diff_EW',-df['ParScore_Diff_NS'],pl.Int16), # used for open-closed room comparisons
#         pl.Series('ExpMaxScore_Diff_EW',-df['ExpMaxScore_Diff_NS'],pl.Float32), # used for open-closed room comparisons
#     )
#     return df


def display_experiments(df):

    if 'Room' in df.columns and df['Room'].n_unique() == 2 and 'Open' in df['Room'].unique() and 'Closed' in df['Room'].unique():
        st.info("Following are WIP experiments showing comparative statistics for Open-Closed room competions. Comparisons include NS vs EW, tricks taken vs DD, par diffs, expected max value diffs.")

        for d in ['NS','EW']:
            g = df.group_by([d[0],d[1],'Room'])
            for k,v in g:
                st.caption(f"Summarize {k[2]} {d} ({k[0]}-{k[1]}) ParScore_Diff_{d}")
                sql_query = f"SUMMARIZE SELECT ParScore_Diff_{d}, DDTricks_Diff, EV_MaxScore_Diff_{d} FROM self WHERE Room='{k[2]}'" # DDTicks is directionally invariant
                ShowDataFrameTable(df, query=sql_query, key=f"display_experiments_{d+'_'.join(k)}_summarize_key")

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


def ShowDataFrameTable(df, key, query='SELECT * FROM self', show_sql_query=True):
    if show_sql_query and st.session_state.show_sql_query:
        st.text(f"SQL Query: {query}")

    # if query doesn't contain 'FROM ', add 'FROM self ' to the beginning of the query. issue is for non-self tables such as exploded_auctions_df.
    # can't just check for startswith 'from self'. Not universal because 'from self' can appear in subqueries or after JOIN.
    # this syntax makes easy work of adding FROM but isn't compatible with polars SQL. duckdb only.
    if 'from ' not in query.lower():
        query = 'FROM self ' + query

    # polars SQL has so many issues that it's impossible to use. disabling until 2030.
    # try:
    #     # First try using Polars SQL. However, Polars doesn't support some SQL functions: string_agg(), agg_value(), some joins are not supported.
    #     if True: # workaround issued by polars. CASE WHEN AVG() ELSE AVG() -> AVG(CASE WHEN ...)
    #         result_df = st.session_state.con.execute(query).pl()
    #     else:
    #         result_df = df.sql(query) # todo: enforce FROM self for security concerns?
    # except Exception as e:
    #     try:
    #         # If Polars fails, try DuckDB
    #         print(f"Polars SQL failed. Trying DuckDB: {e}")
    #         result_df = st.session_state.con.execute(query).pl()
    #     except Exception as e2:
    #         st.error(f"Both Polars and DuckDB SQL engines have failed. Polars error: {e}, DuckDB error: {e2}. Query: {query}")
    #         return None
    
    try:
        result_df = st.session_state.con.execute(query).pl()
        if show_sql_query and st.session_state.show_sql_query:
            st.text(f"Result is a dataframe of {len(result_df)} rows.")
        streamlitlib.ShowDataFrameTable(result_df, key) # requires pandas dataframe.
    except Exception as e:
        st.error(f"duckdb exception: error:{e} query:{query}")
        return None
    
    return result_df


def app_info():
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in Streamlit. AI API is OpenAI. Data engine is Pandas. Query engine is Duckdb. Chat UI uses streamlit-chat. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/PBN_Postmortem_Chatbot. Game data scraped from PBN file.")
    # obsolete when chat was removed: Default AI model:{DEFAULT_AI_MODEL} OpenAI client:{openai.__version__} fastai:{fastai.__version__} safetensors:{safetensors.__version__} sklearn:{sklearn.__version__} torch:{torch.__version__} 
    st.caption(
        f"App:{st.session_state.app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} duckdb:{duckdb.__version__} numpy:{np.__version__} polars:{pl.__version__} Query Params:{st.query_params.to_dict()}")


from fsspec.utils import infer_storage_options

def get_url_protocol(path):
    # Use fsspec to infer the storage type
    options = infer_storage_options(path)
    # Check if the protocol is 'file', which indicates a local file
    return options['protocol']


def chat_input_on_submit():
    prompt = st.session_state.main_prompt_chat_input_key
    ShowDataFrameTable(st.session_state.df, query=prompt, key='user_query_main_doit_key')


def sample_count_on_change():
    st.session_state.single_dummy_sample_count = st.session_state.single_dummy_sample_count_number_input
    change_game_state()


def show_sql_query_change():
    # toggle whether to show sql query
    st.session_state.show_sql_query = st.session_state.sql_query_checkbox


def change_game_state_LIN(file_data,url,path_url,boards,df,everything_df):
    #st.error(f"Unsupported file type: {path_url.suffix}")
    boards = lin.loads(file_data)
    return boards


# Written mostly by chatgpt.
# Define the recursive flattening function
def flatten_df(df):
    """
    Recursively flattens a Polars DataFrame by unnesting struct fields
    and exploding lists, handling any level of nested lists and structs.
    
    Parameters:
    - df: Polars DataFrame with potentially nested data
    
    Returns:
    - Fully flattened Polars DataFrame with no lists or structs
    """

    # Iterate through all columns in the DataFrame
    for col_name in df.columns:
        col = df[col_name]
        dtype = col.dtype

        # If the column is a Struct, unnest it
        if dtype.base_type() == pl.Struct:
            # Get the fields of the struct
            fields = col.struct.fields
            # Unnest each field into a new column
            for field in fields:
                new_col_name = f"{col_name}_{field}"
                df = df.with_columns(col.struct.field(field).alias(new_col_name))
            # Drop the original struct column
            assert col_name in df.columns, col_name
            df = df.drop(col_name)
            return flatten_df(df) # only recurse if df is changed
        
        # If the column is a List, get the first element and check if it is a Struct, if so, unnest it.
        elif dtype.base_type() == pl.List:
            # unnest the inner struct
            print('List:', col_name, dtype.inner, col[0] is None, dtype, dtype.base_type())
            if df.height:
                if col[0] is not None:
                    if col[0].dtype.base_type() == pl.Struct:
                        # confused here: why does concat need to be done after unnest? I thought it unnests in place and drops the original column.
                        df = pl.concat([df,col[0].struct.unnest()],how='horizontal')
                        return flatten_df(df.drop(col_name)) # only recurse if df is changed
    
    return df


def change_game_state_JSON(file_data,url,path_url,boards,df,everything_df):
    st.error(f"Unsupported file type: {path_url.suffix}")
    return None
    boards = json.loads(file_data)
    return boards


def change_game_state_PBN(file_data,url,path_url,boards,df,everything_df):
    if boards is None and df is None:
        with st.spinner("Parsing PBN file ..."):
            boards = pbn.loads(file_data)
            if len(boards) == 0:
                st.warning(f"{url} has no boards.")
                return
    if st.session_state.save_intermediate_files:
        boards_url = pathlib.Path(path_url.stem+'_boards').with_suffix('.pkl')
        boards_path = pathlib.Path(boards_url)
        with st.spinner(f"Saving {boards_url} file ..."):
            with open(boards_path, 'wb') as f:
                pickle.dump(boards, f)
            st.caption(f"Saved {boards_url}. File length is {boards_path.stat().st_size} bytes.")
    return boards


def change_game_state():

    with st.spinner(f'Preparing Bridge Game Postmortem Report. Takes 2 minutes total...'):
        #with st.session_state.chat_container:
        reset_game_data() # wipe out all game state data
        st.session_state.session_id = 'unknown session'
        st.session_state.group_id = 'unknown group'
        st.session_state.player_id = 'unknown player'
        st.session_state.partner_id = 'unknown partner'
        # todo: temp - extract these from data!!!!!!!!!!!!
        st.session_state.pair_direction = 'NS'
        st.session_state.opponent_pair_direction = 'EW'

        url = st.session_state.create_sidebar_text_input_url_key.strip()
        st.text(f"Selected URL: {url}") # using protocol:{get_url_protocol(url)}")

        if url is None or url == '' or (get_url_protocol(url) == 'file' and ('/' in url and '\\' in url and '&' in url)):
            return

        path_url = pathlib.Path(url)
        boards = None
        df = None
        everything_df = None
        # todo: only local intermediate files implemented. is it possible to access them using a url? it gets complicated.
        if url.endswith('_boards.pkl'):
            if not path_url.exists():
                st.warning(f"{url} does not exist.")
                return
            with st.spinner(f"Loading {url} ..."):
                with open(path_url, 'rb') as f:
                    boards = pickle.load(f)
            url = url.replace('_boards.pkl','')
            path_url = pathlib.Path(url)
            Process_PBN(boards,df,everything_df,path_url)
        elif url.endswith('_df.pkl'):
            if not path_url.exists():
                st.warning(f"{url} does not exist.")
                return
            with st.spinner(f"Loading {url} ..."):
                with open(path_url, 'rb') as f:
                    df = pickle.load(f)
            url = url.replace('_df.pkl','')
            path_url = pathlib.Path(url)
            Process_PBN(boards,df,everything_df,path_url)
        elif url.endswith('_everythingdf.parquet'):
            if not path_url.exists():
                st.warning(f"{url} does not exist.")
                return
            with st.spinner(f"Loading {url} ..."):
                everything_df = pl.read_parquet(path_url)
            url = url.replace('_everythingdf.parquet','')
            path_url = pathlib.Path(url)
            Process_PBN(boards,df,everything_df,path_url)
        else:
            with st.spinner(f"Loading {url} ..."):
                try:
                    of = fsspec.open(url, mode='r', encoding='utf-8')
                    with of as f:
                        match path_url.suffix.lower():
                            case '.pbn':
                                file_data = f.read()
                                boards = change_game_state_PBN(file_data,url,path_url,boards,df,everything_df)
                            case '.lin':
                                file_data = f.read()
                                boards = change_game_state_LIN(file_data,url,path_url,boards,df,everything_df)
                            case '.json':
                                file_data = f.read()
                                json_data = json.loads(file_data)
                                json_df = pl.DataFrame(json_data)
                                df = flatten_df(json_df)
                                st.dataframe(df)
                                return
                                #pass
                                # b = boards.unnest('Matches')
                                # pl.DataFrame(b['Sessions'][0].struct.unnest())
                                #boards = change_game_state_JSON(file_data,url,path_url,boards,df,everything_df)
                            case _:
                                st.error(f"Unsupported file type: {path_url.suffix}")
                                return
                except Exception as e:
                    st.error(f"Error opening or reading {url}: {e}")
                    return
        if boards is None:
            st.error(f"Unimplemented file type: {path_url.suffix}")
            return # not yet implemented

        st.session_state.df = Process_PBN(path_url,boards,df,everything_df)
        st.session_state.df = filter_dataframe(st.session_state.df, st.session_state.group_id, st.session_state.session_id, st.session_state.player_id, st.session_state.partner_id)
        assert st.session_state.df.select(pl.col(pl.Object)).is_empty(), f"Found Object columns: {[col for col, dtype in st.session_state.df.schema.items() if dtype == pl.Object]}"
        st.session_state.con.register(st.session_state.con_register_name, st.session_state.df) # ugh, df['scores_l'] must be previously dropped otherwise this hangs. reason unknown.

    return


# this version of perform_hand_augmentations_locked() uses self for class compatibility, older versions did not.
def perform_hand_augmentations_queue(self, hand_augmentation_work):
    return streamlitlib.perform_queued_work(self, hand_augmentation_work, "Hand analysis")


def augment_df(df):
    with st.spinner('Augmenting data...'):
        augmenter = AllAugmentations(df,None,sd_productions=st.session_state.single_dummy_sample_count,progress=st.progress(0),lock_func=perform_hand_augmentations_queue)
        df, hrs_cache_df = augmenter.perform_all_augmentations()
    # with st.spinner('Creating hand data...'):
    #     augmenter = HandAugmenter(df,{},sd_productions=st.session_state.single_dummy_sample_count,progress=st.progress(0),lock_func=perform_hand_augmentations_queue)
    #     df = augmenter.perform_hand_augmentations()
    # with st.spinner('Augmenting with result data...'):
    #     augmenter = ResultAugmenter(df,{})
    #     df = augmenter.perform_result_augmentations()
    # with st.spinner('Augmenting with contract data...'):
    #     augmenter = ScoreAugmenter(df)
    #     df = augmenter.perform_score_augmentations()
    # with st.spinner('Augmenting with DD and SD data...'):
    #     augmenter = DDSDAugmenter(df)
    #     df = augmenter.perform_dd_sd_augmentations()
    # with st.spinner('Augmenting with matchpoints and percentages data...'):
    #     augmenter = MatchPointAugmenter(df)
    #     df = augmenter.perform_matchpoint_augmentations()
    return df


def Process_PBN(path_url,boards,df,everything_df,hrs_d={}):
    with st.spinner("Creating dataframe ..."):
        df = mlBridgeEndplayLib.endplay_boards_to_df({path_url:boards})
        st.dataframe(df) # todo: temp!!!!!!!!!!!
        df = mlBridgeEndplayLib.convert_endplay_df_to_mlBridge_df(df)
        #st.write("After endplay_boards_to_df")
        #ShowDataFrameTable(df, key=f"process_endplay_boards_to_df_key")
    pmb = PBNResultsCalculator()
    df = augment_df(df)
    #st.write("After Perform_DD_SD_Augmentations")
    #ShowDataFrameTable(df, key=f"Perform_DD_SD_Augmentations_key")
    # with st.spinner("Creating Bidding Tables. Very slow. Takes 12 minutes ..."): # todo: make faster. update message.
    #     expression_evaluator = mlBridgeBiddingLib.ExpressionEvaluator()
    #     df = expression_evaluator.create_bidding_table_df(df,st.session_state.bt_prior_bids_to_bt_entry_d)
    # with st.spinner("Creating Ai Auctions ..."):
    #     finder = mlBridgeBiddingLib.AuctionFinder(st.session_state.bt_prior_bids_to_bt_entry_d,st.session_state.bt_bid_to_next_bids_d,st.session_state.exprStr_to_exprID_d)
    #     st.session_state.exprs_dfs_d = finder.augment_df_with_bidding_info(df)
    #     assert len(st.session_state.exprs_dfs_d) == len(df)
    #     for k,expr_df in st.session_state.exprs_dfs_d.items():
    #         print(k,expr_df)
    #         break

    # if save_intermediate_files:
    #     # save df as pickle because it contains object columns. later, they're dropped when creating pbn_df.
    #     df_url = pathlib.Path(path_url.stem+'_df').with_suffix('.pkl')
    #     df_path = pathlib.Path(df_url)
    #     with st.spinner(f"Saving {df_url} file ..."):
    #         with open(df_path, 'wb') as f:
    #             pickle.dump(df, f)
    #     st.caption(f"Saved {df_url}. File length is {df_path.stat().st_size} bytes.")

    # exclude_columns = ['deal','_dealer','_vul','auction','play','_contract'] # drop obsolete columns or object data types. some of these may have been dropped earlier
    # column_order = ['Date','Scoring','Board','Room','Deal','North','East','South','West','Dealer','Vul','Auction','Contract','Play','Score','Claimed','Event','Site','BCFlags']
    # column_order = [c for c in column_order if c in df.columns]
    # # add any not-well-known columns but prepend with underscore to avoid conflicts
    # for c in df.columns:
    #     if c not in column_order:
    #         if c not in exclude_columns:
    #             custom_c = 'Custom_'+c
    #             df = df.rename({c:custom_c})
    #             column_order.append(custom_c)
    # df = df.select(pl.col(column_order))

    # if save_intermediate_files:
    #     everythingdf_url = pathlib.Path(path_url.stem+'_everythingdf').with_suffix('.parquet')
    #     everythingdf_path = pathlib.Path(everythingdf_url)
    #     with st.spinner(f"Saving {everythingdf_url} file ..."):
    #         everything_df.write_parquet(everythingdf_path)
    #     st.caption(f"Saved {everythingdf_url}. File length is {everythingdf_path.stat().st_size} bytes.")

    #display_experiments(df)

    #non_unique_columns_df = df[['index', 'passout', 'trump', 'PBN', 'Hand_N', 'Hand_E', 'Hand_S', 'Hand_W', 'Event', 'BCFlags', 'Room', 'Score', 'bid_type', 'bid_denom', 'bid_penalty', 'bid_level', 'bid_alertable', 'bid_announcement', 'play_rank', 'play_suit', 'Board', 'Dealer', 'Vul', 'iVul', 'Vul_NS', 'Vul_EW', 'Contract', 'BidLvl', 'BidSuit', 'Dbl', 'Declarer_Direction', 'Result', 'Tricks', 'Player_Name_N', 'Player_Name_E', 'Player_Name_S', 'Player_Name_W', 'Player_ID_N', 'Player_ID_E', 'Player_ID_S', 'Player_ID_W']]
    #st.dataframe(non_unique_columns_df)
    return df


def filter_dataframe(df, group_id, session_id, player_id, partner_id):
    # First filter for sessions containing player_id

    # df = df.filter(
    #     pl.col('group_id').eq(group_id) &
    #     pl.col('session_id').eq(session_id)
    # )
    
    # Set direction variables based on where player_id is found
    player_direction = None
    if player_id in df['Player_ID_N']:
        player_direction = 'N'
        partner_direction = 'S'
        pair_direction = 'NS'
        opponent_pair_direction = 'EW'
    elif player_id in df['Player_ID_E']:
        player_direction = 'E'
        partner_direction = 'W'
        pair_direction = 'EW'
        opponent_pair_direction = 'NS'
    elif player_id in df['Player_ID_S']:
        player_direction = 'S'
        partner_direction = 'N'
        pair_direction = 'NS'
        opponent_pair_direction = 'EW'
    elif player_id in df['Player_ID_W']:
        player_direction = 'W'
        partner_direction = 'E'
        pair_direction = 'EW'
        opponent_pair_direction = 'NS'

    # todo: not sure what to do here. pbns might not contain names or ids. endplay has names but not ids.
    if player_direction is None:
        df = df.with_columns(
            pl.lit(True).alias('Boards_I_Played'), # player_id could be numeric
            pl.lit(True).alias('Boards_I_Declared'), # player_id could be numeric
            pl.lit(True).alias('Boards_Partner_Declared'), # partner_id could be numeric
        )
    else:
        # Store in session state
        st.session_state.player_direction = player_direction
        st.session_state.partner_direction = partner_direction
        st.session_state.pair_direction = pair_direction
        st.session_state.opponent_pair_direction = opponent_pair_direction

        # Columns used for filtering to a specific player_id and partner_id. Needs multiple with_columns() to unnest overlapping columns.
        df = df.with_columns(
            pl.col(f'Player_ID_{player_direction}').eq(pl.lit(str(player_id))).alias('Boards_I_Played'), # player_id could be numeric
            pl.col('Declarer_ID').eq(pl.lit(str(player_id))).alias('Boards_I_Declared'), # player_id could be numeric
            pl.col('Declarer_ID').eq(pl.lit(str(partner_id))).alias('Boards_Partner_Declared'), # partner_id could be numeric
        )
    df = df.with_columns(
        pl.col('Boards_I_Played').alias('Boards_We_Played'),
        pl.col('Boards_I_Played').alias('Our_Boards'),
        (pl.col('Boards_I_Declared') | pl.col('Boards_Partner_Declared')).alias('Boards_We_Declared'),
    )
    df = df.with_columns(
        (pl.col('Boards_I_Played') & ~pl.col('Boards_We_Declared') & pl.col('Contract').ne('PASS')).alias('Boards_Opponent_Declared'),
    )

    return df

def create_sidebar():
    st.sidebar.caption('Build:'+st.session_state.app_datetime)

    # example valid urls
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
    #default_url = 'file://c:/sw/bridge/ML-Contract-Bridge/src/Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://c:\sw/bridge\ML-Contract-Bridge\src\Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    default_url = 'DDS_Camrose24_1- BENCAM22 v WBridge5.pbn' #'1746217537-NzYzEYivsA@250502.PBN' # '3494191054-1682343601-bsalita.lin'
    #default_url = 'GIB-Thorvald-8638-2024-08-23.pbn'
    st.sidebar.text_input('Enter URL:', default_url, on_change=change_game_state, key='create_sidebar_text_input_url_key', help='Enter a URL or pathless local file name.') # , on_change=change_game_state
    # using css to change button color for the entire button width. The color was choosen to match the the restrictive text colorizer (:green-background[Go]) used in st.info() below.
    css = """section[data-testid="stSidebar"] div.stButton button {
        background-color: rgba(33, 195, 84, 0.1);
        width: 50px;
        }"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.sidebar.button('Go', on_click=change_game_state, key='create_sidebar_go_button_key', help='Load PBN data from URL.')

    if st.session_state.player_id is None:
        return

    st.session_state.pdf_link = st.sidebar.empty()

    with st.sidebar.expander('Developer Settings', False):
        # do not use .sidebar in expander. it's already in the sidebar.
        # SELECT Board, Vul, ParContract, ParScore_NS, Custom_ParContract FROM self
        st.checkbox('Show SQL Query',value=st.session_state.show_sql_query,key='sql_query_checkbox',on_change=show_sql_query_change,help='Show SQL used to query dataframes.')
        # These files are reloaded each time for development purposes. Only takes a second.
        # todo: put filenames into a .json or .toml file?
        st.session_state.single_dummy_sample_count = st.number_input('Single Dummy Sample Count',value=st.session_state.single_dummy_sample_count,key='single_dummy_sample_count_number_input',on_change=sample_count_on_change,min_value=1,max_value=1000,step=1,help='Number of random deals to generate for calculating single dummy probabilities. Larger number (10 to 30) is more accurate but slower. Use 1 to 5 for fast, less accurate results.')

    st.session_state.default_favorites_file = pathlib.Path(
        'default.favorites.json')
    st.session_state.player_id_custom_favorites_file = pathlib.Path(
        'favorites/'+st.session_state.player_id+'.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path(
        'favorites/debug.favorites.json')
    read_configs()
    return

# todo: put this in PBNResultsCalculator class?
def read_configs():

    st.session_state.default_favorites_file = pathlib.Path(
        'default.favorites.json')
    st.session_state.player_id_custom_favorites_file = pathlib.Path(
        f'favorites/{st.session_state.player_id}.favorites.json')
    st.session_state.debug_favorites_file = pathlib.Path(
        'favorites/debug.favorites.json')

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
        st.session_state.favorites = favorites
        #st.session_state.vetted_prompts = get_vetted_prompts_from_favorites(favorites)
    else:
        st.session_state.favorites = None

    if st.session_state.player_id_custom_favorites_file.exists():
        with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
            player_id_favorites = json.load(f)
        st.session_state.player_id_favorites = player_id_favorites
    else:
        st.session_state.player_id_favorites = None

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
        st.session_state.debug_favorites = debug_favorites
    else:
        st.session_state.debug_favorites = None

    # display missing prompts in favorites
    if 'missing_in_summarize' not in st.session_state:
        # Get the prompts from both locations
        summarize_prompts = st.session_state.favorites['Buttons']['Summarize']['prompts']
        vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']

        # Process the keys to ignore leading '@'
        st.session_state.summarize_keys = {p.lstrip('@') for p in summarize_prompts}
        st.session_state.vetted_keys = set(vetted_prompts.keys())

        # Find items in summarize_prompts but not in vetted_prompts. There should be none.
        st.session_state.missing_in_vetted = st.session_state.summarize_keys - st.session_state.vetted_keys
        assert len(st.session_state.missing_in_vetted) == 0, f"Oops. {st.session_state.missing_in_vetted} not in {st.session_state.vetted_keys}."

        # Find items in vetted_prompts but not in summarize_prompts. ok if there's some missing.
        st.session_state.missing_in_summarize = st.session_state.vetted_keys - st.session_state.summarize_keys

        print("\nItems in Vetted_Prompts but not in Summarize.prompts:")
        for item in st.session_state.missing_in_summarize:
            print(f"- {item}: {vetted_prompts[item]['title']}")
    return


# todo: use this similar to bridge_game_postmortem_streamlit.py
def reset_game_data():

    # Default values for session state variables
    reset_defaults = {
        'game_description_default': None,
        'group_id_default': None,
        'session_id_default': None,
        'section_name_default': None,
        'player_id_default': None,
        'partner_id_default': None,
        'player_name_default': None,
        'partner_name_default': None,
        'player_direction_default': None,
        'partner_direction_default': None,
        'pair_id_default': None,
        'pair_direction_default': None,
        'opponent_pair_direction_default': None,
    }
    
    # Initialize default values if not already set
    for key, value in reset_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize additional session state variables that depend on defaults.
    reset_session_vars = {
        'df': None,
        'game_description': st.session_state.game_description_default,
        'group_id': st.session_state.group_id_default,
        'session_id': st.session_state.session_id_default,
        'section_name': st.session_state.section_name_default,
        'player_id': st.session_state.player_id_default,
        'partner_id': st.session_state.partner_id_default,
        'player_name': st.session_state.player_name_default,
        'partner_name': st.session_state.partner_name_default,
        'player_direction': st.session_state.player_direction_default,
        'partner_direction': st.session_state.partner_direction_default,
        'pair_id': st.session_state.pair_id_default,
        'pair_direction': st.session_state.pair_direction_default,
        'opponent_pair_direction': st.session_state.opponent_pair_direction_default,
        #'sidebar_loaded': False,
        'analysis_started': False,   # new flag for analysis sidebar rewrite
        'vetted_prompts': [],
        'pdf_assets': [],
        'sql_query_mode': False,
        'sql_queries': [],
        'game_urls_d': {},
        'tournament_session_urls_d': {},
    }
    
    for key, value in reset_session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

    return


def initialize_website_specific():

    st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
    st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/master/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.
    st.session_state.game_results_url_default = None
    st.session_state.game_name = 'pbn'
    st.session_state.game_results_url = st.session_state.game_results_url_default
    # todo: put filenames into a .json or .toml file?
    st.session_state.rootPath = pathlib.Path('e:/bridge/data')
    #st.session_state.acblPath = st.session_state.rootPath.joinpath('acbl')
    #st.session_state.favoritesPath = pathlib.joinpath('favorites'),
    st.session_state.savedModelsPath = st.session_state.rootPath.joinpath('SavedModels')

    streamlit_chat.message(
        f"Hi. I'm Morty. Your friendly postmortem chatbot. I only want to chat about {st.session_state.game_name} pair matchpoint games using a Mitchell movement and not shuffled.", key='intro_message_1', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm optimized for large screen devices such as a notebook or monitor. Do not use a smartphone.", key='intro_message_2', logo=st.session_state.assistant_logo)
    #streamlit_chat.message(
    #    f"To start our postmortem chat, I'll need an {st.session_state.game_name} player number. I'll use it to find player's latest {st.session_state.game_name} club game. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        f"To start our postmortem chat, I'll need a PBN file or URL. It will be the subject of our chat.", key='intro_message_3', logo=st.session_state.assistant_logo)
    #streamlit_chat.message(
    #    f"Enter any {st.session_state.game_name} player number in the left sidebar.", key='intro_message_4', logo=st.session_state.assistant_logo)
    streamlit_chat.message(
        "I'm just a Proof of Concept so don't double me.", key='intro_message_5', logo=st.session_state.assistant_logo)
    return


# todo: this class should be universal. its methods should initialize generic values followed by call outs to app-specific methods.
class PBNResultsCalculator(PostmortemBase):
    """PBN Results Calculator Streamlit application."""
    
    def __init__(self):
        super().__init__()
        # App-specific initialization
    
    # App-specific methods
    def parse_pbn_file(self, pbn_content):
        """Parse PBN file content."""
        # Implementation for parsing PBN files
        pass
    
    def calculate_results(self, pbn_data):
        """Calculate results from PBN data."""
        # Implementation for calculating results
        pass
    
    def export_results(self, results, format="csv"):
        """Export results in various formats."""
        # Implementation for exporting results
        pass
    
    def file_uploader_callback(self):
        """Handle file upload events."""
        # Implementation
        pass
    
    # Override abstract methods
    def initialize_session_state(self):
        """Initialize app-specific session state."""

        # todo: obsolete these in preference to 
        # App-specific session state
        if 'pbn_file' not in st.session_state:
            st.session_state.pbn_file = None
        if 'pbn_data' not in st.session_state:
            st.session_state.pbn_data = None
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'player_id' not in st.session_state:
            st.session_state.player_id = 'Unknown'
        if 'recommended_board_max' not in st.session_state:
            st.session_state.recommended_board_max = 100
        if 'save_intermediate_files' not in st.session_state:
            st.session_state.save_intermediate_files = False

        st.set_page_config(layout="wide")
        # Add this auto-scroll code
        streamlitlib.widen_scrollbars()

        if platform.system() == 'Windows': # ugh. this hack is required because torch somehow remembers the platform where the model was created. Must be a bug. Must lie to torch.
            pathlib.PosixPath = pathlib.WindowsPath
        else:
            pathlib.WindowsPath = pathlib.PosixPath
        
        if 'player_id' in st.query_params:
            player_id = st.query_params['player_id']
            if not isinstance(player_id, str):
                st.error(f'player_id must be a string {player_id}')
                st.stop()
            st.session_state.player_id = player_id
        else:
            st.session_state.player_id = None

        first_time_defaults = {
            'first_time': True,
            'single_dummy_sample_count': 10,
            'show_sql_query': True, # os.getenv('STREAMLIT_ENV') == 'development',
            'use_historical_data': False,
            'do_not_cache_df': True, # todo: set to True for production
            'con': duckdb.connect(), # IMPORTANT: duckdb.connect() hung until previous version was installed.
            'con_register_name': 'self',
            'main_section_container': st.empty(),
            'app_datetime': datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z'),
            'current_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for key, value in first_time_defaults.items():
            st.session_state[key] = value

        self.reset_game_data()
        self.initialize_website_specific()
        return


    def reset_game_data(self):
        """Reset game data."""
        # Implementation
        reset_game_data()

    def initialize_website_specific(self):
        """Initialize app-specific components."""
        # Implementation
        initialize_website_specific()
    
    def process_prompt_macros(self, sql_query):
        """Process app-specific prompt macros."""
        # First process standard macros
        sql_query = super().process_prompt_macros(sql_query)
        # Then app-specific macros
        # Implementation
        return sql_query
    
    # Customize standard methods as needed
    def create_sidebar(self):
        """Create app-specific sidebar."""
        # Call super method for standard elements
        create_sidebar() # accessing the global function, not the class method.

    def create_main_content(self):
        """Create app-specific main content."""
        # Implementation
        st.title("PBN Results Calculator")
        
        # File upload section
        st.header("Upload PBN File")
        uploaded_file = st.file_uploader("Choose a PBN file", type="pbn", on_change=self.file_uploader_callback)
        
        if st.session_state.pbn_data is not None:
            # Display PBN data
            st.header("PBN Data")
            st.dataframe(st.session_state.pbn_data, use_container_width=True)
            
            # Calculate button
            if st.button("Calculate Results"):
                results = self.calculate_results(st.session_state.pbn_data)
                st.session_state.results = results
        
        if st.session_state.results is not None:
            # Results section
            st.header("Results")
            self.ShowDataFrameTable(
                st.session_state.results, 
                "results_table",
                show_sql_query=False
            )
            
            # Export options
            st.header("Export")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV"):
                    self.export_results(st.session_state.results, "csv")
            with col2:
                if st.button("Export as PDF"):
                    self.export_results(st.session_state.results, "pdf")

    # todo: copied from acbl_postmortem_streamlit.py
    def write_report(self):
        # bar_format='{l_bar}{bar}' isn't working in stqdm. no way to suppress r_bar without editing stqdm source code.
        # todo: need to pass the Button title to the stqdm description. this is a hack until implemented.
        st.session_state.main_section_container = st.container(border=True)
        with st.session_state.main_section_container:
            report_title = f"Bridge Game Postmortem Report Personalized for {st.session_state.player_name}" # can't use (st.session_state.player_id) because of href link below.
            report_creator = f"Created by https://{st.session_state.game_name}.postmortem.chat"
            report_event_info = f"{st.session_state.game_description} (event id {st.session_state.session_id})."
            report_game_results_webpage = f"Results Page: {st.session_state.game_results_url}"
            report_your_match_info = f"Your pair was {st.session_state.pair_id}{st.session_state.pair_direction} in section {st.session_state.section_name}. You played {st.session_state.player_direction}. Your partner was {st.session_state.partner_name} ({st.session_state.partner_id}) who played {st.session_state.partner_direction}."
            st.markdown(f"### {report_title}")
            st.markdown(f"##### {report_creator}")
            st.markdown(f"#### {report_event_info}")
            st.markdown(f"##### {report_game_results_webpage}")
            st.markdown(f"#### {report_your_match_info}")
            pdf_assets = st.session_state.pdf_assets
            pdf_assets.clear()
            pdf_assets.append(f"# {report_title}")
            pdf_assets.append(f"#### {report_creator}")
            pdf_assets.append(f"### {report_event_info}")
            pdf_assets.append(f"#### {report_game_results_webpage}")
            pdf_assets.append(f"### {report_your_match_info}")
            st.session_state.button_title = 'Summarize' # todo: generalize to all buttons!
            selected_button = st.session_state.favorites['Buttons'][st.session_state.button_title]
            vetted_prompts = st.session_state.favorites['SelectBoxes']['Vetted_Prompts']
            sql_query_count = 0
            for stats in stqdm(selected_button['prompts'], desc='Creating personalized report...'):
                assert stats[0] == '@', stats
                stat = vetted_prompts[stats[1:]]
                for i, prompt in enumerate(stat['prompts']):
                    if 'sql' in prompt and prompt['sql']:
                        #print('sql:',prompt["sql"])
                        if i == 0:
                            streamlit_chat.message(f"Morty: {stat['help']}", key=f'morty_sql_query_{sql_query_count}', logo=st.session_state.assistant_logo)
                            pdf_assets.append(f"### {stat['help']}")
                        prompt_sql = prompt['sql']
                        sql_query = self.process_prompt_macros(prompt_sql) # we want the default process_prompt_macros() to be used.
                        query_df = ShowDataFrameTable(st.session_state.df, query=sql_query, key=f'sql_query_{sql_query_count}')
                        if query_df is not None:
                            pdf_assets.append(query_df)
                        sql_query_count += 1

            # As a text link
            #st.markdown('[Back to Top](#your-personalized-report)')

            # As an html button (needs styling added)
            # can't use link_button() restarts page rendering. markdown() will correctly jump to href.
            # st.link_button('Go to top of report',url='#your-personalized-report')\
            report_title_anchor = report_title.replace(' ','-').lower()
            st.markdown(f'<a target="_self" href="#{report_title_anchor}"><button>Go to top of report</button></a>', unsafe_allow_html=True)

        if st.session_state.pdf_link.download_button(label="Download Personalized Report",
                data=streamlitlib.create_pdf(st.session_state.pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
                file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
                disabled = len(st.session_state.pdf_assets) == 0,
                mime='application/octet-stream',
                key='personalized_report_download_button'):
            st.warning('Personalized report downloaded.')
        return


    # todo: copied from acbl_postmortem_streamlit.py
    def ask_sql_query(self):

        if st.session_state.show_sql_query:
            with st.container():
                with bottom():
                    st.chat_input('Enter a SQL query e.g. SELECT PBN, Contract, Result, N, S, E, W', key='main_prompt_chat_input_key', on_submit=chat_input_on_submit)


if __name__ == "__main__":
    if 'first_time' not in st.session_state: # todo: change to 'app' not in st.session_state
        st.session_state.app = PBNResultsCalculator()
    st.session_state.app.main() 