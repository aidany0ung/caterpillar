import requests as req
import pandas as pd
import json

def pubChemQuery(input,type):
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
    url += 'compound/'+type+'/' + input + '/property/ExactMass,XLogP,Complexity,Charge,HBondDonorCount,HBondAcceptorCount/JSON'
    r = req.get(url)
    return r.json()

def parseQuery(json):
    return json['PropertyTable']['Properties'][0]

df = pd.read_csv('../data/bppp.csv', sep=',', header=0)

if 'ExactMass' not in df.columns:
    df['ExactMass'] = -1

# For each smile molecule, query PubChem for the molecular properties and add each key, valye from parseQuery to the dataframe
for i in range(len(df)):
    # Ensure no extra requests are made
    if df['ExactMass'].iloc[i] != -1:
        print('triggered')
        continue

    # Make the request and parse the response, otherwise throw error
    try:
        query = pubChemQuery(df['smile'].iloc[i],'smiles')
        for key, value in parseQuery(query).items():
            df.loc[i, key] = value
    except:
        try:
            query = pubChemQuery(df['smile'].iloc[i],'name')
            for key, value in parseQuery(query).items():
                df.loc[i, key] = value
        except:
            print('Error with smile: ' + df['smile'].iloc[i])

    # Progressively update the file
    if i % 50 == 0:
        print('Finished ' + str(i) + ' molecules')
        df.to_csv('../data/bppp.csv')
    print(i)

df.to_csv('../data/bppp.csv')