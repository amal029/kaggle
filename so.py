#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt


def app(r, col):
    ret = 0
    tt = [x.strip().lower() for x in r.split(';')]
    if col in tt:
        ret += 1
    return ret


def main(f):
    df = pd.read_csv(f)
    dfi = df[['Country', 'IDE']].groupby('Country')
    dfic = dfi['IDE'].count().dropna().sort_values(ascending=False)[:15]
    # Just the top 15 countries from where we have significant responses
    df = df[df['Country'].isin(dfic.index.values)]

    dfh = df[df['Professional'] == 'Professional developer']
    dfh = dfh[['HoursPerWeek', 'Country']].groupby('Country')
    dfh = dfh['HoursPerWeek'].mean().dropna().sort_values(ascending=False)

    # Now let us find out which IDE is most popular
    dfi = df[['Country', 'IDE']]
    dfi = dfi.dropna()

    ides = set()
    for i in dfi['IDE']:
        ides.update([x.strip().lower() for x in i.split(';')])

    for i in ides:
        dfi[i] = 0
    for i in ides:
        dfi[i] += dfi['IDE'].apply(lambda x: app(x, i))

    dfi = dfi.groupby('Country')
    cs = [x for x, _ in dfi]

    h = {c: None for c in cs}
    m = {c: 0 for c in cs}

    for j in ides:
        mm = dfi[j].mean()
        for c in cs:
            if m[c] < mm[c]:
                m[c] = mm[c]
                h[c] = j
    # print(pd.DataFrame(data=list(h.values()),
    #                    index=cs))

    # Now let's look at version control
    # Most popular version control
    pvc = df.groupby('VersionControl')['VersionControl'].count()
    pvct = pvc.sum()
    pvs = pvc.index
    for i in pvs:
        pvc[i] = (pvc[i]/pvct)*100  # In percentage

    cc = df.groupby('Country')['Country'].count()
    cs = cc.index

    vc = df[['VersionControl', 'Country']].groupby(['Country',
                                                    'VersionControl'])
    vc = vc['VersionControl'].count().sort_values(ascending=False)
    # print(vc)

    pvc.sort_values(ascending=False)[:5].plot(kind='barh')
    # dfh[:10].plot(kind='barh', legend=False)
    plt.show()


if __name__ == '__main__':
    plt.style.use('ggplot')
    main('./so_survey_results_public.csv.zip')
