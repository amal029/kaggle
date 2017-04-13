#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt


def app(r, tutu):
    ret = 0
    if tutu is not None:
        if r['Religeon.1.Name'] == '1.'+tutu:
            ret += r['Religeon.1.Population']
        if r['Religeon.2.Name'] == '2.'+tutu:
            ret += r['Religeon.2.Population']
        if r['Religeon.3.Name'] == '3.'+tutu:
            ret += r['Religeon.3.Population']
    else:
        if not (r['Religeon.1.Name'] == '1.Hindus' or
                r['Religeon.1.Name'] == '1.Muslims' or
                r['Religeon.1.Name'] == '1.Christians'):
            ret += r['Religeon.1.Population']
        if not (r['Religeon.2.Name'] == '2.Hindus' or
                r['Religeon.2.Name'] == '2.Muslims' or
                r['Religeon.2.Name'] == '2.Christians'):
            ret += r['Religeon.2.Population']
        if not (r['Religeon.3.Name'] == '3.Hindus' or
                r['Religeon.3.Name'] == '3.Muslims' or
                r['Religeon.3.Name'] == '3.Christians'):
            ret += r['Religeon.3.Population']

    return ret


def explore(f):
    df = pd.read_csv(f)

    # Now something else
    # print(df['Religeon.3.Name'][:20])
    # Christians

    df['Total'] = (df['Religeon.1.Population'] +
                   df['Religeon.2.Population'] + df['Religeon.3.Population'])
    df['Hindus'] = df.apply(lambda x: app(x, 'Hindus'), axis=1)
    df['Muslims'] = df.apply(lambda x: app(x, 'Muslims'), axis=1)
    df['Christians'] = df.apply(lambda x: app(x, 'Christians'), axis=1)
    df['Others'] = df.apply(lambda x: app(x, None), axis=1)

    df_states = df.groupby(['State'])
    rels = df_states[['Hindus', 'Muslims', 'Christians',
                      'Others', 'Total']].sum()
    rels['Hindus'] = (rels['Hindus'] / rels['Total']) * 100.0

    rels['Muslims'] = (rels['Muslims'] / rels['Total']) * 100.0

    rels['Christians'] = (rels['Christians'] / rels['Total']) * 100.0

    rels['Others'] = (rels['Others'] / rels['Total']) * 100.0
    rels = rels[['Hindus', 'Muslims', 'Christians', 'Others']]

    # ax = rels.plot(kind='bar')
    # ax.set_xticklabels(rels.index, rotation='vertical')

    # Literacy rate/state.
    lit = df_states[['Males..Literate', 'Females..Literate',
                     'Persons..literate', 'Total']].sum()
    lit['popu_lit_rate'] = lit['Persons..literate']/lit['Total']*100
    lit['male_lit_rate'] = (lit['Males..Literate'] /
                            lit['Persons..literate']*100)
    lit['female_lit_rate'] = (lit['Females..Literate'] /
                              lit['Persons..literate']*100)
    lit = lit[['popu_lit_rate', 'female_lit_rate', 'male_lit_rate']]
    # lit = lit[['male_lit_rate', 'female_lit_rate',
    #            'popu_lit_rate']].join(rels)
    lit1 = lit[['popu_lit_rate']].join(rels)
    print(lit1.sort_values(by='popu_lit_rate', ascending=False))
    lit2 = lit[['male_lit_rate']].join(rels)
    lit3 = lit[['female_lit_rate']].join(rels)
    lit1.plot(kind='bar')
    lit2.plot(kind='bar')
    lit3.plot(kind='bar')
    # set_xticklabels(lit.index, rotation='vertical')

    plt.show()


def gt1(f):
    df = pd.read_csv(f, encoding="ISO-8859-1")

    df_country = df[df['success'] == 1].groupby('country_txt')
    df_country_c = df_country['eventid'].count()
    df_country_c.sort(ascending=False)
    ax = df_country_c[:10].plot(kind='barh', grid=True)
    ax.set_ylabel('Country')
    ax.set_xlabel('Number of successful terror attacks')
    ax.get_figure().savefig('/tmp/top10.png', bbox_inches='tight')


def gt2(f):
    df = pd.read_csv(f, encoding="ISO-8859-1")

    f_india_tot = df[df['country_txt'] == 'India'].groupby('iyear').count()
    f_pak_tot = df[df['country_txt'] == 'Pakistan'].groupby('iyear').count()

    # India
    df_india = df[(df['country_txt'] == 'India')
                  & (df['success'] == 1)]
    df_india_year = df_india.groupby('iyear')
    f_india = df_india_year.count()

    # Pakistan
    df_pak = df[(df['country_txt'] == 'Pakistan')
                & (df['success'] == 1)]
    df_pak_year = df_pak.groupby('iyear')

    f_pak = df_pak_year.count()
    f_countries = pd.DataFrame(index=f_india.index)
    f_countries['India'] = f_india['eventid']
    f_countries['Pakistan'] = f_pak['eventid']
    axi = f_countries.plot(kind='bar', grid=True)
    axi.set_ylabel('Number of successful terror attacks')
    axi.set_xlabel('Year')
    axi.set_xticklabels(f_countries.index.map(lambda x: "'"+str(x)[2:]))
    axi.get_figure().savefig('/tmp/india_pak.png', bbox_inches='tight')

    f_eff = pd.DataFrame(index=f_india.index)
    f_eff['India_stopped'] = ((f_india_tot['eventid'] - f_india['eventid'])
                              / f_india_tot['eventid'])*100
    print(f_eff['India_stopped'].values.mean())
    f_eff['Pak_stopped'] = ((f_pak_tot['eventid'] - f_pak['eventid'])
                            / f_pak_tot['eventid'])*100
    axi = f_eff.plot(kind='bar', grid=True)
    axi.set_xlabel('Year')
    axi.set_ylabel('% attacks stopped')
    axi.set_xticklabels(f_eff.index.map(lambda x: "'"+str(x)[2:]))
    axi.get_figure().savefig('/tmp/india_pak_stopped.png', bbox_inches='tight')


if __name__ == '__main__':
    plt.style.use('ggplot')
    # explore('india_census_district.csv')
    gt1('globalterrorismdb_0616dist.csv')
    gt2('globalterrorismdb_0616dist.csv')
