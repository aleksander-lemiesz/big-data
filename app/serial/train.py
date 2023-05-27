import pandas as pd
from sklearn.neighbors import NearestNeighbors
from app.data_utils import create_df, serial_pipeline

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = create_df('../../files/NYPD_short.csv')
    df, orig_df = serial_pipeline(df)

    neighbors = NearestNeighbors(algorithm='ball_tree').fit(df)
    dist, indices = neighbors.kneighbors(df)
    # for j in range(142675):
    for j in range(5):
        print('Crimes similar to crime number ' + str(j))
        for i in range(4):
            if dist[j][i] < 10.0:
                print('Crime' + str(i) + ': ' + str(orig_df.iloc[[indices[j][i]]]))
                print('Distance: ' + str(dist[j][i]) + '\n')
            if i == 4:
                print('\n')
