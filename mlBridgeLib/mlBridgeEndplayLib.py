# Contains functions for:
# 1. reading endplay compatible files
# 2. creates endplay board classes
# 3. creates an endplay polars df from boards classes
# 4. converts the endplay df to a mlBridge df


import polars as pl
import pickle
from collections import defaultdict

import endplay.parsers.lin as lin


def lin_files_to_boards_dict(lin_files_l,boards_d,bbo_lin_files_cache_file=None):
    load_count = 0
    for i,lin_file in enumerate(lin_files_l):
        if i % 10000 == 0:
            print(f'{i}/{len(lin_files_l)} {load_count=} file:{lin_file}')
        if lin_file in boards_d:
            continue
        with open(lin_file, 'r', encoding='utf-8') as f:
            try:
                boards_d[lin_file] = lin.load(f)
            except Exception as e:
                print(f'error: {i}/{len(lin_files_l)} file:{lin_file} error:{e}')
                continue
        load_count += 1
        if load_count % 1000000 == 0:
            if bbo_lin_files_cache_file is not None:
                with open(bbo_lin_files_cache_file, 'wb') as f:
                    pickle.dump(boards_d,f)
                print(f"Saved {str(bbo_lin_files_cache_file)}: len:{len(boards_d)} size:{bbo_lin_files_cache_file.stat().st_size}")
    return boards_d


def endplay_boards_to_df(boards_d): 
    # Initialize data dictionary as defaultdict(list)
    board_d = defaultdict(list)

    # There's always only one board per lin file.(?). You can have multiple boards by simply concatenating lin files with '\n' in between them.
    for lin_file,boards_in_lin_file in boards_d.items():
        for i,b in enumerate(boards_in_lin_file):
            board_d['board_num'].append(b.board_num)
            board_d['dealer'].append(None if b._dealer is None else b._dealer.name) # None if passed out
            board_d['vulnerability'].append(None if b._dealer is None else b._vul.name) # None if passed out, weird
            board_d['passout'].append(b._contract.is_passout() if b._contract else None)
            board_d['contract'].append(str(b._contract) if b._contract else None)
            board_d['level'].append(b._contract.level if b._contract else None)
            board_d['denom'].append(b._contract.denom.name if b._contract else None)
            board_d['trump'].append(b.deal.trump.name)
            board_d['penalty'].append(b._contract.penalty.name if b._contract else None)
            board_d['declarer'].append(b._contract.declarer.name if b._contract else None)
            board_d['result'].append(b._contract.result if b._contract else None)
            board_d['score'].append(b._contract.score(b._vul) if b._contract else None)
            board_d['claimed'].append(b.claimed)
            board_d['PBN'].append(b.deal.to_pbn())
            board_d['Hand_N'].append(str(b.deal.north))
            board_d['Hand_E'].append(str(b.deal.east))
            board_d['Hand_S'].append(str(b.deal.south))
            board_d['Hand_W'].append(str(b.deal.west))
            
            # todo: dangerously assumes every boards's info has same keys as first board's info.
            custom_naming_exceptions = {
                'BCFlags':'BCFlags',
                'Date':'Date',
                'Event':'Event',
                'Room':'Room',
                'Score':'Score',
                'Scoring':'Scoring',
                'Site':'Site',
                'North':'Player_N',
                'East':'Player_E',
                'South':'Player_S',
                'West':'Player_W'
            }
            if i == 0: # first board
                custom_c_l = set(b.info.keys()).difference(boards_d.keys())
                print(f'Creating columns for custom info keys: {custom_c_l}')
            else: # subsequent boards must match first board's info keys
                custom_c_diffs = set(b.info.keys()).symmetric_difference(custom_c_l)
                assert custom_c_diffs == set(), custom_c_diffs
            for c in custom_c_l:
                board_d[custom_naming_exceptions.get(c,'Custom_'+c)].append(b.info[c])

            board_d['source_file'].append(str(lin_file)) # todo: change key to be str instead of pathlib.Path?
            bid_type = []
            denom = []
            penalty = []
            level = []
            alertable = []
            announcement = []
            for bid in b.auction:
                if hasattr(bid, 'denom'):
                    bid_type.append('Contract')
                    denom.append(bid.denom.name)
                    penalty.append(None)
                    level.append(bid.level)
                    alertable.append(bid.alertable)
                    announcement.append(bid.announcement)
                else:
                    bid_type.append('Penalty')
                    denom.append(None)
                    penalty.append(bid.penalty.name)
                    level.append(None)
                    alertable.append(bid.alertable)
                    announcement.append(bid.announcement)
            board_d['bid_type'].append(bid_type)
            board_d['bid_denom'].append(denom)
            board_d['bid_penalty'].append(penalty)
            board_d['bid_level'].append(level)
            board_d['bid_alertable'].append(alertable)
            board_d['bid_announcement'].append(announcement)
            play_rank = []
            play_suit = []
            for play in b.play:
                play_rank.append(play.rank.name)
                play_suit.append(play.suit.name)
            board_d['play_rank'].append(play_rank)
            board_d['play_suit'].append(play_suit)
    
    # create polars df
    df = pl.DataFrame(board_d)
    return df


# convert lin file columns to conform to bidding table columns.

EpDealer_to_Dealer_d = {
    'north':'N',
    'east':'E',
    'south':'S',
    'west':'W'
}

EpDenom_to_SHDCN_d = {
    'spades':'S',
    'hearts':'H',
    'diamonds':'D',
    'clubs':'C',
    'nt':'N'
}

EpVulnerability_to_Vul_d = {
    'none': 'None',
    'ns': 'N_S',
    'ew': 'E_W',
    'both': 'Both'
}

EpVulnerability_to_iVul_d = {
    'none': 0,
    'ns': 1,
    'ew': 2,
    'both': 3
}

EpVulnerability_to_Vul_NS_Bool_d = {
    'none': False,
    'ns': True,
    'ew': False,
    'both': True
}

EpVulnerability_to_Vul_EW_Bool_d = {
    'none': False,
    'ns': False,
    'ew': True,
    'both': True
}

EpContract_to_Contract_d = {
    '♠':'S',
    '♥':'H',
    '♦':'D',
    '♣':'C',
    'NT':'N',
    'Pass':'PASS'
}

EpPenalty_to_Dbl_d = {
    'passed':'',
    'doubled':'x',
    'redoubled':'xx'
}


def convert_endplay_df_to_mlBridge_df(df):

    # todo: create Date column using date embedded in source file name.
    # todo: is pair number ns, pair number ew needed?

    df = df.with_columns(
        pl.Series('Board',df['board_num'],pl.UInt8),
        pl.Series('Dealer', df['dealer'].replace(EpDealer_to_Dealer_d), pl.String),# categorical?
        pl.Series('Vul',df['vulnerability'].replace(EpVulnerability_to_Vul_d),pl.String),# categorical, yes
        pl.Series('iVul',df['vulnerability'].replace(EpVulnerability_to_iVul_d),pl.UInt8), # categorical, yes
        pl.col('vulnerability').replace_strict(EpVulnerability_to_Vul_NS_Bool_d,return_dtype=pl.Boolean).alias('Vul_NS'),
        pl.col('vulnerability').replace_strict(EpVulnerability_to_Vul_EW_Bool_d,return_dtype=pl.Boolean).alias('Vul_EW'),
        #pl.Series('passout',df['passout'],pl.Boolean), # todo: make passout a boolean in previous step.
        # first NT->N and suit symbols to SHDCN
        pl.Series('Contract',df['contract'],pl.String).str.replace('NT','N').str.replace('♠','S').str.replace('♥','H').str.replace('♦','D').str.replace('♣','C'),
        pl.Series('BidLvl',df['level'].cast(pl.UInt8, strict=False),pl.UInt8), # todo: make level a uint8 in previous step.
        pl.Series('BidSuit',df['denom'].replace(EpDenom_to_SHDCN_d),pl.String),# categorical, yes
        pl.Series('trump',df['trump'].replace(EpDenom_to_SHDCN_d),pl.String),# categorical?
        pl.Series('Dbl',df['penalty'].replace(EpPenalty_to_Dbl_d),pl.String),# categorical, yes
        pl.Series('Declarer_Direction', df['declarer'].replace(EpDealer_to_Dealer_d), pl.String),# categorical, yes
        pl.Series('Result',df['result'].cast(pl.Int8, strict=False).fill_nan(0),pl.Int8),
        pl.Series('Tricks',df['level'].cast(pl.Int8, strict=False).fill_nan(0)+df['result'].cast(pl.Int8, strict=False).fill_nan(0)+6,pl.UInt8),
        pl.Series('Score',df['score'].cast(pl.Int16, strict=False).fill_nan(0),pl.Int16),
        df['claimed'].cast(pl.Boolean, strict=False),
        pl.Series('Player_Name_N',df['Player_N'],pl.String),
        pl.Series('Player_Name_E',df['Player_E'],pl.String),
        pl.Series('Player_Name_S',df['Player_S'],pl.String),
        pl.Series('Player_Name_W',df['Player_W'],pl.String),
        pl.Series('Player_ID_N',df['Player_N'],pl.String), # todo: fake player id
        pl.Series('Player_ID_E',df['Player_E'],pl.String), # todo: fake player id
        pl.Series('Player_ID_S',df['Player_S'],pl.String), # todo: fake player id
        pl.Series('Player_ID_W',df['Player_W'],pl.String), # todo: fake player id
        pl.Series('Pair_Number_NS',pl.lit(0),pl.String), # todo: fake Pair_Number_NS
        pl.Series('Pair_Number_EW',pl.lit(0),pl.String), # todo: fake Pair_Number_EW
        pl.Series('source_file',df['source_file'],pl.String),
        pl.Series('bid_type',df['bid_type'],pl.List(pl.String)),# categorical?
        pl.Series('bid_denom',df['bid_denom'],pl.List(pl.String)),# categorical? #.replace(denom_to_SHDCN_d)
        pl.Series('bid_penalty',df['bid_penalty'],pl.List(pl.String)),# categorical? #.replace(penalty_to_Dbl_d)
        #pl.Series('bid_level',df['bid_level'].cast(pl.List(pl.Int64), strict=False)+1,pl.List(pl.Int64)), # todo: make bid_level a uint8 in previous step.
        pl.Series('bid_alertable',df['bid_alertable'],pl.List(pl.Boolean)),
        pl.Series('bid_announcement',df['bid_announcement'],pl.List(pl.String)),
        pl.Series('play_rank',df['play_rank'],pl.List(pl.String)),# categorical?
        pl.Series('play_suit',df['play_suit'],pl.List(pl.String)),# categorical?
    )
    # drop unused or obsolete columns
    df = df.drop(
        {
            'board_num',
            'dealer',
            'vulnerability',
            'contract',
            'level',
            'denom',
            'penalty',
            'declarer',
            'result',
            'score',
            'Player_N',
            'Player_E',
            'Player_S',
            'Player_W',
        }
    )
    return df


