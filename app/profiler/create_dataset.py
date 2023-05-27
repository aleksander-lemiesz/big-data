from app.data_utils import create_df, nn_pipeline

if __name__ == '__main__':
    full = 'full'
    df = create_df(f'../../files/NYPD_{full}.csv')
    df = nn_pipeline(df)
    df.to_csv(f'data/data_{full}.csv', index=False)
