from scripts.etl import load_data_to_df, preprocess_data, spark
from scripts.eda import EDAAnalyzer
from scripts.feng import *
from pprint import pprint



if __name__ == '__main__':
    """
    This script performs the ETL, EDA, FENG and MODELLING processes on the data and prints the results.
    """
    ################### ETL ###################

    # Directorio de datos
    data_dir = 'data/'
    # Carga los datos en un solo dataframe
    df = load_data_to_df(data_dir)
    # Preprocesa los datos
    df = preprocess_data(df)

    ################### EDA ###################

    eda = EDAAnalyzer(df)
    try:
        results = eda.load_results('output/eda_results.pkl')
    except FileNotFoundError:
        results = eda.perform_complete_eda()
        # Save EDA results to disk
        eda.save_results(results, 'output/eda_results.pkl')
    # Pretty print the EDA results
    print("Data Quality Analysis:\n")
    pprint(results['data_quality']['null_counts'].T)
    print("\nCategorical Analysis:\n")
    for col, analysis in results['categorical_analysis'].items():
        print(f"Column: {col}")
        pprint(analysis)
        print()
    print("\nNumerical Analysis:\n")
    for col, analysis in results['numerical_analysis'].items():
        print(f"Column: {col}")
        pprint(analysis.T)
        print()
    print("\nCorrelation Matrix:\n")
    print(results['correlation_matrix'])

    df_clients = eda.return_unique_clients_df()

    ################### FENG ###################

    try:
        df_clients = spark.read.parquet('output/clients_df')
    except:
        df_clients = feature_engineering(df, df_clients)
        # fill na in avg_transaction_amount and income_expense_ratio with 0
        df_clients = df_clients.fillna(0, subset=["avg_transaction_amount", "income_expense_ratio"])
        save_clients_df(df_clients, 'output/clients_df')

    print("\nClients DataFrame:")
    df_clients.show()








