
# streamlit program to display Bridge game deal statistics from a PBN file.
# Invoke from system prompt using: streamlit run CalculatePBNResults_Streamlit.py

import streamlit as st
from streamlit_extras.bottom_container import bottom
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
                sql_query = f"SUMMARIZE SELECT ParScore_Diff_{d}, DDTricks_Diff, EV_MaxScore_Diff_{d} FROM df WHERE Room='{k[2]}'" # DDTicks is directionally invariant
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


def ShowDataFrameTable(df, key, query='SELECT * FROM df', show_sql_query=True):
    if show_sql_query and st.session_state.show_sql_query:
        st.text(f"SQL Query: {query}")
    try:
        # todo: could implement duckdb.execute(query) to implement multiple statements. duckdb.sql(query) only works for one select statements.
        df = duckdb.sql(query).pl()
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

    # show completed dataframe. Limit columns to, say 1000, to avoid overwhelming the dataframe rendering UI.
    ShowDataFrameTable(df, query='SELECT PBN, Hand_N, Suit_N_S, Contract, Result, Score_NS, ParScore_NS, ParScore_Diff_NS, EV_NS_Max, EV_NS_MaxCol, EV_MaxScore_Diff_NS FROM df', key=f"Show_Sample_Dataframe_key")

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


def create_sidebar():
    st.sidebar.caption('Build:'+app_datetime)

    # example valid urls
    #default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1-%20BENCAM22%20v%20WBridge5.pbn'
    #default_url = 'file://c:/sw/bridge/ML-Contract-Bridge/src/Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://c:\sw/bridge\ML-Contract-Bridge\src\Calculate_PBN_Results/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    #default_url = r'file://DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
    default_url = 'https://raw.githubusercontent.com/BSalita/Calculate_PBN_Results/master/DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'
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

    # SELECT Board, Vul, ParContract, ParScore_NS, Custom_ParContract FROM df
    st.sidebar.checkbox('Show SQL Query',value=show_sql_query_default,key='create_sidebar_show_sql_query_checkbox_key',on_change=sql_query_on_change,help='Show SQL used to query dataframes.')


if __name__ == '__main__':

    # first time only defaults
    if 'first_time_only_initialized' not in st.session_state:
        st.session_state.first_time_only_initialized = True
        st.set_page_config(layout="wide")
        streamlitlib.widen_scrollbars()

    # Refreshable defaults
    app_datetime = datetime.fromtimestamp(pathlib.Path(__file__).stat().st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
    #pbn_filename_default = 'DDS_Camrose24_1- BENCAM22 v WBridge5.pbn'  # local filename
    single_dummy_sample_count_default = 2  # number of random deals to generate for calculating single dummy probabilities. Use smaller number for testing.
    st.session_state.single_dummy_sample_count = single_dummy_sample_count_default
    show_sql_query_default = True
    st.session_state.show_sql_query = show_sql_query_default
    save_intermediate_files = False # leave False for now. saving intermediate files presents problems with persistance. where to do it? how to clean up? how to handle multiple users?
    recommended_board_max = 10000

    create_sidebar()

    if 'df' not in st.session_state:
        st.title("PBN Statistics Generator and Query Engine")
        app_info()
        st.info("*Start by entering a URL and clicking the :green-background[Go] button on the left sidebar.* The process takes 1 to 2 minutes to complete. When the running man in the top right corner stops, the data is ready for query.")
    else:
        default_sql_query = f"SELECT PBN, Hand_N, Suit_N_S, Contract, Result, Score_NS, ParScore_NS, ParScore_Diff_NS, EV_NS_Max, EV_NS_MaxCol, EV_MaxScore_Diff_NS FROM df"
        with bottom():
            st.caption(f"Enter a SQL query in the box below. e.g. {default_sql_query}")
            st.chat_input('Enter a SQL query', key='main_prompt_chat_input_key', on_submit=chat_input_on_submit)

