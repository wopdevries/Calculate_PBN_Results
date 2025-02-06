
# streamlit program to display Bridge game deal statistics from a PBN file.
# Invoke from system prompt using: streamlit run CalculatePBNResults_Streamlit.py

import streamlit as st
import streamlit_chat
from streamlit_extras.bottom_container import bottom
from stqdm import stqdm

import pathlib
import fsspec
import polars as pd # used for __version__ only
import polars as pl
import duckdb
import pickle
from collections import defaultdict
from datetime import datetime, timezone
import sys

import endplay # for __version__
from endplay.parsers import pbn, lin, json
from endplay.types import Deal, Contract, Denom, Player, Penalty, Vul
from endplay.dds import par, calc_all_tables
from endplay.dealer import generate_deals

sys.path.append(str(pathlib.Path.cwd().joinpath('streamlitlib')))  # global
sys.path.append(str(pathlib.Path.cwd().joinpath('mlBridgeLib')))  # global

import streamlitlib
#import mlBridgeLib
import mlBridgeAugmentLib
import mlBridgeEndplayLib
import mlBridgeBiddingLib


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
    st.caption(f"Project lead is Robert Salita research@AiPolice.org. Code written in Python. UI written in streamlit. Data engine is polars. Query engine is duckdb. Bridge lib is endplay. Self hosted using Cloudflare Tunnel. Repo:https://github.com/BSalita/Calculate_PBN_Results")
    st.caption(
        f"App:{app_datetime} Python:{'.'.join(map(str, sys.version_info[:3]))} Streamlit:{st.__version__} Pandas:{pd.__version__} polars:{pl.__version__} endplay:{endplay.__version__} Query Params:{st.query_params.to_dict()}")


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
    st.session_state.single_dummy_sample_count = st.session_state.create_sidebar_single_dummy_sample_count_key
    if 'df' in st.session_state:
        LoadPage()


def sql_query_on_change():
    st.session_state.show_sql_query = st.session_state.create_sidebar_show_sql_query_checkbox_key
    if 'df' in st.session_state:
        LoadPage()


def LoadPage_LIN(file_data,url,path_url,boards,df,everything_df):
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


def LoadPage_JSON(file_data,url,path_url,boards,df,everything_df):
    st.error(f"Unsupported file type: {path_url.suffix}")
    return None
    boards = json.loads(file_data)
    return boards


def LoadPage_PBN(file_data,url,path_url,boards,df,everything_df):
    if boards is None and df is None:
        with st.spinner("Parsing PBN file ..."):
            boards = pbn.loads(file_data)
            if len(boards) == 0:
                st.warning(f"{url} has no boards.")
                return
            if len(boards) > recommended_board_max:
                st.warning(f"{url} has {len(boards)} boards. More than {recommended_board_max} boards may result in instability.")
    if save_intermediate_files:
        boards_url = pathlib.Path(path_url.stem+'_boards').with_suffix('.pkl')
        boards_path = pathlib.Path(boards_url)
        with st.spinner(f"Saving {boards_url} file ..."):
            with open(boards_path, 'wb') as f:
                pickle.dump(boards, f)
            st.caption(f"Saved {boards_url}. File length is {boards_path.stat().st_size} bytes.")
    return boards


def LoadPage():

    #with st.session_state.chat_container:

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
                    match path_url.suffix:
                        case '.pbn':
                            file_data = f.read()
                            boards = LoadPage_PBN(file_data,url,path_url,boards,df,everything_df)
                        case '.lin':
                            file_data = f.read()
                            boards = LoadPage_LIN(file_data,url,path_url,boards,df,everything_df)
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
                            #boards = LoadPage_JSON(file_data,url,path_url,boards,df,everything_df)
                        case _:
                            st.error(f"Unsupported file type: {path_url.suffix}")
                            return
            except Exception as e:
                st.error(f"Error opening or reading {url}: {e}")
                return
    if boards is None:
        st.error(f"Unimplemented file type: {path_url.suffix}")
        return # not yet implemented
    Process_PBN(path_url,boards,df,everything_df)
    return


# todo: implement stqdm progress bar
def Process_PBN(path_url,boards,df,everything_df,hrs_d={}):
    with st.spinner("Converting PBN to Endplay Dataframe ..."):
        df = mlBridgeEndplayLib.endplay_boards_to_df({path_url:boards})
        #st.write("After endplay_boards_to_df")
        #ShowDataFrameTable(df, key=f"process_endplay_boards_to_df_key")
    with st.spinner("Converting Endplay Dataframe to mlBridge Dataframe ..."):
        df = mlBridgeEndplayLib.convert_endplay_df_to_mlBridge_df(df)
        #st.write("After convert_endplay_df_to_mlBridge_df_key")
        #ShowDataFrameTable(df, key=f"process_convert_endplay_df_to_mlBridge_df_key")
    with st.spinner("Performing Hand Augmentations ..."):
        df = mlBridgeAugmentLib.perform_hand_augmentations(df,hrs_d,sd_productions=st.session_state.single_dummy_sample_count)
        #st.write("After perform_hand_augmentations")
        #ShowDataFrameTable(df, key=f"perform_hand_augmentations_key")
    with st.spinner("Performing Matchpoint and Percentage Augmentations ..."):
        df = mlBridgeAugmentLib.PerformMatchPointAndPercentAugmentations(df)
        #st.write("After PerformMatchPointAndPercentAugmentations")
        #ShowDataFrameTable(df, key=f"PerformMatchPointAndPercentAugmentations_key")
    with st.spinner("Performing Result Augmentations ..."):
        df = mlBridgeAugmentLib.PerformResultAugmentations(df,hrs_d)
        #st.write("After PerformResultAugmentations")
        #ShowDataFrameTable(df, key=f"PerformResultAugmentations_key")
    with st.spinner("Performing DD_SD Augmentations ..."):
        df = mlBridgeAugmentLib.Perform_DD_SD_Augmentations(df)
        #st.write("After Perform_DD_SD_Augmentations")
        #ShowDataFrameTable(df, key=f"Perform_DD_SD_Augmentations_key")
    with st.spinner("Creating Bidding Tables. Very slow. Takes 12 minutes ..."): # todo: make faster. update message.
        expression_evaluator = mlBridgeBiddingLib.ExpressionEvaluator()
        df = expression_evaluator.create_bidding_table_df(df,st.session_state.bt_prior_bids_to_bt_entry_d)
    with st.spinner("Creating Ai Auctions ..."):
        finder = mlBridgeBiddingLib.AuctionFinder(st.session_state.bt_prior_bids_to_bt_entry_d,st.session_state.bt_bid_to_next_bids_d,st.session_state.exprStr_to_exprID_d)
        st.session_state.exprs_dfs_d = finder.augment_df_with_bidding_info(df)
        assert len(st.session_state.exprs_dfs_d) == len(df)
        for k,expr_df in st.session_state.exprs_dfs_d.items():
            print(k,expr_df)
            break

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

    st.session_state.df = df


def filter_dataframe(df, group_id, session_id, player_id, partner_id):
    # First filter for sessions containing player_id

    df = df.filter(
        pl.col('group_id').eq(group_id) &
        pl.col('session_id').eq(session_id)
    )
    
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
    st.sidebar.caption('Build:'+app_datetime)

    # example valid urls
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
    #default_url = 'file://c:/sw/bridge/ML-Contract-Bridge/src/Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://c:\sw/bridge\ML-Contract-Bridge\src\Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    default_url = '3494191054-1682343601-bsalita.lin'
    #default_url = 'GIB-Thorvald-8638-2024-08-23.pbn'
    st.sidebar.text_input('Enter URL:', default_url, on_change=LoadPage, key='create_sidebar_text_input_url_key', help='Enter a URL or pathless local file name.') # , on_change=LoadPage
    # using css to change button color for the entire button width. The color was choosen to match the the restrictive text colorizer (:green-background[Go]) used in st.info() below.
    css = """section[data-testid="stSidebar"] div.stButton button {
        background-color: rgba(33, 195, 84, 0.1);
        width: 50px;
        }"""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    st.sidebar.button('Go', on_click=LoadPage, key='create_sidebar_go_button_key', help='Load PBN data from URL.')

    st.session_state.single_dummy_sample_count = st.sidebar.number_input('Single Dummy Sample Count',value=single_dummy_sample_count_default,key='create_sidebar_single_dummy_sample_count_key',on_change=sample_count_on_change,min_value=1,max_value=1000,step=1,help='Number of random deals to generate for calculating single dummy probabilities. Larger number (10 to 30) is more accurate but slower. Use 1 to 5 for fast, less accurate results.')

    # SELECT Board, Vul, ParContract, ParScore_NS, Custom_ParContract FROM self
    st.sidebar.checkbox('Show SQL Query',value=show_sql_query_default,key='create_sidebar_show_sql_query_checkbox_key',on_change=sql_query_on_change,help='Show SQL used to query dataframes.')


def read_favorites():

    if st.session_state.default_favorites_file.exists():
        with open(st.session_state.default_favorites_file, 'r') as f:
            favorites = json.load(f)
            st.session_state.favorites = favorites

    if st.session_state.player_id_custom_favorites_file.exists():
        with open(st.session_state.player_id_custom_favorites_file, 'r') as f:
            player_id_favorites = json.load(f)
            st.session_state.player_id_favorites = player_id_favorites

    if st.session_state.debug_favorites_file.exists():
        with open(st.session_state.debug_favorites_file, 'r') as f:
            debug_favorites = json.load(f)
            st.session_state.debug_favorites = debug_favorites


def load_vetted_prompts(json_file):

    sql_queries = []
    if json_file.exists():
        with open(json_file) as f:
            json_data = json.load(f)
        
        # Navigate the JSON path to get the appropriate list of prompts
        vetted_prompts = [json_data['SelectBoxes']['Vetted_Prompts'][p[1:]] for p in json_data["Buttons"]['Summarize']['prompts']]
    
    return vetted_prompts


def process_prompt_macros(sql_query):
    replacements = {
        '{Player_Direction}': st.session_state.player_direction,
        '{Partner_Direction}': st.session_state.partner_direction,
        '{Pair_Direction}': st.session_state.pair_direction,
        '{Opponent_Pair_Direction}': st.session_state.opponent_pair_direction
    }
    for old, new in replacements.items():
        sql_query = sql_query.replace(old, new)
    return sql_query


def show_dfs(vetted_prompts, pdf_assets):
    sql_query_count = 0

    # bar_format='{l_bar}{bar}' isn't working in stqdm. no way to suppress r_bar without editing stqdm source code.
    for category in stqdm(list(vetted_prompts), desc='Morty is analyzing your game...', bar_format='{l_bar}{bar}'): #[:-3]:
        #print('category:',category)
        if "prompts" in category:
            for i,prompt in enumerate(category["prompts"]):
                #print('prompt:',prompt) 
                if "sql" in prompt and prompt["sql"]:
                    if i == 0:
                        streamlit_chat.message(f"Morty: {category['help']}", key=f'morty_sql_query_{sql_query_count}', logo=st.session_state.assistant_logo)
                        pdf_assets.append(f"## {category['help']}")
                    #print('sql:',prompt["sql"])
                    prompt_sql = prompt['sql']
                    sql_query = process_prompt_macros(prompt_sql)
                    query_df = ShowDataFrameTable(st.session_state.df, query=sql_query, key=f'sql_query_{sql_query_count}')
                    if query_df is not None:
                        pdf_assets.append(query_df)
                    sql_query_count += 1
                    #break
        #break


if __name__ == '__main__':

    # first time only defaults
    if 'first_time_only_initialized' not in st.session_state:
        st.session_state.first_time_only_initialized = True
        st.set_page_config(layout="wide")
        streamlitlib.widen_scrollbars()

        st.session_state.assistant_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/main/assets/logo_assistant.gif?raw=true' # 🥸 todo: put into config. must have raw=true for github url.
        st.session_state.guru_logo = 'https://github.com/BSalita/Bridge_Game_Postmortem_Chatbot/blob/main/assets/logo_guru.png?raw=true' # 🥷todo: put into config file. must have raw=true for github url.

        # Create connection
        st.session_state.con = duckdb.connect()
    
        rootPath = pathlib.Path('e:/bridge/data')
        acblPath = rootPath.joinpath('acbl')
        bboPath = rootPath.joinpath('bbo')
        dataPath = bboPath.joinpath('data')
    
        # takes 10s
        bbo_eval_bidding_tables_d_filename = 'bbo_eval_bidding_tables_d.pkl'
        bbo_eval_bidding_tables_d_file = dataPath.joinpath(bbo_eval_bidding_tables_d_filename)
        with open(bbo_eval_bidding_tables_d_file, 'rb') as f:
            st.session_state.bt_prior_bids_to_bt_entry_d, st.session_state.bt_bid_to_next_bids_d = pickle.load(f)
        print(f"Loaded {bbo_eval_bidding_tables_d_filename}: size:{bbo_eval_bidding_tables_d_file.stat().st_size}")

        # takes 1s per 1m rows.
        # todo: evaluated_expressions_d is probably obsolete as its concatenated into train_df.
        bbo_eval_bidding_tables_d_filename = 'bbo_bidding_tables_dicts.pkl'
        bbo_eval_bidding_tables_d_file = dataPath.joinpath(bbo_eval_bidding_tables_d_filename)
        with open(bbo_eval_bidding_tables_d_file, 'rb') as f:
            #evaluated_expressions_d, exprStr_to_exprID_d = pickle.load(f)
            _, st.session_state.exprStr_to_exprID_d = pickle.load(f)
        print(f"Loaded {bbo_eval_bidding_tables_d_filename}: size:{bbo_eval_bidding_tables_d_file.stat().st_size}")
        #print(f"{len(evaluated_expressions_d)=} {len(exprStr_to_exprID_d)=}")
        print(f"{len(st.session_state.exprStr_to_exprID_d)=}")

    # Refreshable defaults
    app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    #pbn_filename_default = 'DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'  # local filename
    single_dummy_sample_count_default = 2  # number of random deals to generate for calculating single dummy probabilities. Use smaller number for testing.
    st.session_state.single_dummy_sample_count = single_dummy_sample_count_default
    show_sql_query_default = True
    st.session_state.show_sql_query = show_sql_query_default
    save_intermediate_files = False # leave False for now. saving intermediate files presents problems with persistance. where to do it? how to clean up? how to handle multiple users?
    recommended_board_max = 10000
    
    st.session_state.group_id_default = 0 # numeric or string?
    st.session_state.session_id_default = 0 # numeric or string?
    st.session_state.pair_id_default = None
    st.session_state.player_id_default = 0 # numeric or string?
    st.session_state.partner_id_default = None # numeric or string?
    st.session_state.player_direction_default = 'N'
    st.session_state.partner_direction_default = 'S'
    st.session_state.pair_direction_default = 'NS'
    st.session_state.opponent_pair_direction_default = 'EW'
    st.session_state.game_url_default = None
    st.session_state.game_date_default = 'Unknown date' #pd.to_datetime(st.session_state.df['Date'].iloc[0]).strftime('%Y-%m-%d')

    st.session_state.group_id = st.session_state.group_id_default
    st.session_state.session_id = st.session_state.session_id_default
    st.session_state.pair_id = st.session_state.pair_id_default
    st.session_state.player_id = st.session_state.player_id_default
    st.session_state.partner_id = st.session_state.partner_id_default
    st.session_state.player_direction = st.session_state.player_direction_default
    st.session_state.partner_direction = st.session_state.partner_direction_default
    st.session_state.pair_direction = st.session_state.pair_direction_default
    st.session_state.opponent_pair_direction = st.session_state.opponent_pair_direction_default
    st.session_state.game_url = st.session_state.game_url_default
    st.session_state.game_date = st.session_state.game_date_default
    st.session_state.use_historical_data = False # use historical data from file or get from url

    st.session_state.default_sql_query = "SELECT PBN, Hand_N, Suit_N_S, Board, Contract, Result, Tricks, Score_NS, DDScore_NS, ParScore_NS, ParScore_Diff_NS, EV_NS_Max, EV_NS_MaxCol, EV_MaxScore_Diff_NS FROM self"

    if 'df' in st.session_state:
        create_sidebar()
    else:

        create_sidebar()
        LoadPage()

        # personalize to player, partner, opponents, etc.
        st.session_state.df = filter_dataframe(st.session_state.df, st.session_state.group_id, st.session_state.session_id, st.session_state.player_id, st.session_state.partner_id)

        # Register DataFrame as 'results' view
        st.session_state.con.register('self', st.session_state.df)

        # ShowDataFrameTable(df, key='everything_df_key', query='SELECT Board, Pct_NS, Pct_EW, MP_NS, MP_EW FROM self')

        st.session_state.default_favorites_file = pathlib.Path(
            'default.favorites.json')
        st.session_state.player_id_custom_favorites_file = pathlib.Path(
            f'favorites/{st.session_state.player_id}.favorites.json')
        st.session_state.debug_favorites_file = pathlib.Path(
            'favorites/debug.favorites.json')
        read_favorites()

        # pdf_assets = []
        #pdf_assets.append(f"# Bridge Game Postmortem Report Personalized for {st.session_state.player_id}")
        #pdf_assets.append(f"### Created by https://ffbridge.postmortem.chat")
        #pdf_assets.append(f"## Game Date:? Session:{st.session_state.session_id} Player:{st.session_state.player_id} Partner:{st.session_state.partner_id}")

        # st.session_state.vetted_prompts = load_vetted_prompts(st.session_state.default_favorites_file)

        # with st.container(border=True):
            # st.markdown('### Your Personalized Report')
            # st.text(f'Game Date:? Session:{st.session_state.session_id} Player:{st.session_state.player_id} Partner:{st.session_state.partner_id}')
            # show_dfs(st.session_state.vetted_prompts, pdf_assets)

            # # As a text link
            # #st.markdown('[Back to Top](#your-personalized-report)')

            # # As an html button (needs styling added)
            # # can't use link_button() restarts page rendering. markdown() will correctly jump to href.
            # # st.link_button('Go to top of report',url='#your-personalized-report')
            # st.markdown(''' <a target="_self" href="#your-personalized-report">
            #                     <button>
            #                         Go to top of report
            #                     </button>
            #                 </a>''', unsafe_allow_html=True)

        # if st.sidebar.download_button(label="Download Personalized Report",
        #         data=streamlitlib.create_pdf(pdf_assets, title=f"Bridge Game Postmortem Report Personalized for {st.session_state.player_id}"),
        #         file_name = f"{st.session_state.session_id}-{st.session_state.player_id}-morty.pdf",
        #         mime='application/octet-stream'):
        #     st.warning('Personalized report downloaded.')

            # Register DataFrame as 'results' view
        st.session_state.con.register('self', st.session_state.df)
        ShowDataFrameTable(st.session_state.df, query=st.session_state.default_sql_query, key='show_default_query_key')

        for i,(k,auctions_df) in enumerate(st.session_state.exprs_dfs_d.items()):
            print(i,k,auctions_df)
            if i >= 10:
                break
            ShowDataFrameTable(auctions_df, query='SELECT DISTINCT index, Board, PBN, Dealer, Vul FROM auctions_df', key=f'show_distinct_key_{k}')
            ShowDataFrameTable(auctions_df, query='SELECT * EXCLUDE (Board, PBN, Dealer, Vul) FROM auctions_df', key=f'show_exclude_distincts_key_{k}')
            exploded_auctions_df = auctions_df.explode(['criteria_str','criteria_col','criteria_values'])
            ShowDataFrameTable(exploded_auctions_df, query='SELECT DISTINCT index, Board, PBN, Dealer, Vul FROM exploded_auctions_df', key=f'show_distinct_exploded_auctions_key_{k}')
            ShowDataFrameTable(exploded_auctions_df, query='SELECT * EXCLUDE (Board, PBN, Dealer, Vul) FROM exploded_auctions_df', key=f'show_exclude_distincts_exploded_auctions_key_{k}')

    if st.session_state.show_sql_query:
        with st.container():
            with bottom():
                st.caption(f"Enter a SQL query in the box below. e.g. {st.session_state.default_sql_query}")
                st.chat_input('Enter a SQL query', key='main_prompt_chat_input_key', on_submit=chat_input_on_submit)

