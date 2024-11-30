import polars as pl

def preprocess_tax_df(df):
    """_summary_

    Args:
        df (polars): _description_
    """
    
    #Drop c10 as it is a recoding of the target variable
    df = df.drop('c10')
    
    # Subset to categorical cols
    cat_cols = [col for col in df_train.columns if df_train[col].dtype == pl.String]
    
    # Fill categorical cols missing vals with "missing"
    df_train = df_train.with_columns(
        [pl.col(col).fill_null('missing').alias(col) for col in cat_cols]
    )
   
    
    #Recode column n4 to be catergorical with 999 (missing) being 0 and 1 otherwise
    df_train = df_train.with_columns(pl.when(
    pl.col('n4') == 999).then(0).otherwise(1).alias('n4_recoded'))
    