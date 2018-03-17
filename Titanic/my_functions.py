def df_value_counts(df, params):
    for p in params:
        print(df[p].value_counts())


def test():
    print('hello')
