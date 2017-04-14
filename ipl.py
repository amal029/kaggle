#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt


def concat_match_team(dft, x):
    ret = dft[dft['Team_Id'] == x['Match_Winner_Id']]['Team_Short_Code']
    return ret.values[0]


def concat_man_match(dfp, x):
    ret = dfp[dfp['Player_Id'] == x['Man_Of_The_Match_Id']]['Player_Name']
    return ret.values[0]


def main(files):
    dfbbb = pd.read_csv(files[0])
    dfp = pd.read_csv(files[1])
    dft = pd.read_csv(files[2])
    dfs = pd.read_csv(files[3])
    dfm = pd.read_csv(files[4]).dropna()
    dfpm = pd.read_csv(files[5])

    # Who won the most matches
    dfm['Match_Winner_Name'] = dfm.apply(lambda x: concat_match_team(dft, x),
                                         axis=1)

    dfm['Man_of_Match'] = dfm.apply(lambda x: concat_man_match(dfp, x), axis=1)

    mmw = dfm.groupby('Match_Winner_Name')['Match_Id'].count().sort_values(
        ascending=False)
    mmw.plot(kind='barh')
    plt.show()
    plt.clf()

    # Now plot the most man of the matches
    mmm = dfm.groupby('Man_of_Match')['Match_Id'].count().sort_values(
        ascending=False)[:10]   # Just the top-10
    mmm.plot(kind='barh')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main(['Ball_by_Ball.csv', 'Player.csv',
          'Team.csv', 'Season.csv', 'Match.csv',
          'Player_Match.csv'])
