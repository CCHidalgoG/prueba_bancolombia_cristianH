from pyspark.sql import functions as F
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Class to handle EDA operations for banking fraud detection"""

    def __init__(self, df, spark=None):
        """Initialize with a Spark DataFrame"""
        self.df = df
        self.spark = spark if spark else df.sparkSession
        self.total_rows = None
        self.categorical_cols = [
            'tipo_cli', 'estado_cli', 'nivel_academico', 'profn',
            'ocup', 'tipo_contrato', 'genero_cli', 'tipo_vivienda',
            'pais_nacim', 'operac_moneda_extranjera', 'nivel_riesgo_ciiu',
            'trn_desc_tip_cta', 'trn_efec', 'trn_canal_serv_efec'
        ]
        self.numerical_cols = [
            'ing_mes', 'egresos_mes', 'trn_monto', 'score_riesgo_mun'
        ]

    def analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        logger.info("Starting data quality analysis")
        try:
            self.total_rows = self.df.count()

            # Calculate null counts and percentages
            null_expressions = []
            for c in self.df.columns:
                null_expressions.append(
                    F.sum(F.when(F.col(c).isNull() | F.col(c).contains('None'), 1)
                          .otherwise(0)).alias(c)
                )
            null_counts = self.df.select(null_expressions).toPandas()

            # Calculate basic statistics for numerical columns
            numeric_expressions = []
            for c in self.numerical_cols:
                numeric_expressions.extend([
                    F.count(F.col(c)).alias(c + '_count'),
                    F.round(F.mean(F.col(c)), 2).alias(c + '_mean'),
                    F.round(F.stddev(F.col(c)), 2).alias(c + '_stddev'),
                    F.round(F.min(F.col(c)), 2).alias(c + '_min'),
                    F.round(F.max(F.col(c)), 2).alias(c + '_max')
                ])
            numeric_stats = self.df.select(numeric_expressions).toPandas()

            return {
                'null_counts': null_counts,
                'total_rows': self.total_rows,
                'numeric_stats': numeric_stats
            }
        except Exception as e:
            logger.error(f"Error in analyze_data_quality: {str(e)}")
            raise

    def analyze_categorical_variables(self) -> Dict[str, pd.DataFrame]:
        """Analyze categorical variables"""
        logger.info("Starting categorical variables analysis")
        try:
            categorical_stats = {}

            for col in self.categorical_cols:
                stats = self.df.groupBy(col) \
                    .agg(F.count('*').alias('count')) \
                    .withColumn('percentage',
                                F.round(F.col('count') * 100 / self.total_rows, 2)) \
                    .orderBy('count', ascending=False) \
                    .toPandas()
                categorical_stats[col] = stats

            return categorical_stats

        except Exception as e:
            logger.error(f"Error in categorical analysis: {str(e)}")
            raise

    def analyze_numerical_variables(self) -> Dict[str, pd.DataFrame]:
        """Analyze numerical variables"""
        logger.info("Starting numerical variables analysis")
        try:
            numerical_stats = {}

            # Calculate statistics for each numerical column
            for col in self.numerical_cols:
                stats = self.df.select(
                    F.mean(col).alias('mean'),
                    F.stddev(col).alias('stddev'),
                    F.min(col).alias('min'),
                    F.max(col).alias('max'),
                    F.percentile_approx(col, 0.5).alias('median'),
                    F.skewness(col).alias('skewness'),
                    F.kurtosis(col).alias('kurtosis')
                ).toPandas()
                numerical_stats[col] = stats

            return numerical_stats

        except Exception as e:
            logger.error(f"Error in numerical analysis: {str(e)}")
            raise

    #TODO: Add this to FEENG.py
    def analyze_risk_patterns(self) -> pd.DataFrame:
        """Analyze risk patterns with improved memory management"""
        logger.info("Starting risk pattern analysis")
        try:
            risk_window = Window.partitionBy('mun_res')

            # Create risk analysis DataFrame with optimizations
            risk_analysis = self.df.withColumn(
                'avg_transaction_by_location',
                F.avg('trn_monto').over(risk_window)
            ).withColumn(
                'transaction_frequency',
                F.count('trn_monto').over(risk_window)
            ).withColumn(
                'risk_score',
                F.when(F.col('nivel_riesgo_ciiu').isin(['alto']), 3)
                .when(F.col('nivel_riesgo_ciiu').isin(['medio']), 2)
                .otherwise(1)
            )

            # Add sampling to reduce data size
            sample_fraction = 0.1  # Adjust this value based on your data size
            sampled_risk_analysis = risk_analysis.sample(False, sample_fraction)

            # Select only necessary columns
            columns_needed = ['mun_res', 'avg_transaction_by_location',
                              'transaction_frequency', 'risk_score']
            result = sampled_risk_analysis.select(columns_needed)

            # Cache the result before converting to Pandas
            result.cache()

            return result.toPandas()

        except Exception as e:
            logger.error(f"Error in risk pattern analysis: {str(e)}")
            raise

    def analyze_customer_behavior(self) -> pd.DataFrame:
        """Analyze customer behavior"""
        logger.info("Starting customer behavior analysis")
        try:
            behavior_metrics = self.df.groupBy('cli').agg(
                F.avg('trn_monto').alias('avg_transaction_amount'),
                F.count('trn_monto').alias('transaction_count'),
                F.sum(F.when(F.col('trn_efec') == 'si', 1).otherwise(0))
                .alias('cash_transactions'),
                F.max('ing_mes').alias('monthly_income'),
                F.max('egresos_mes').alias('monthly_expenses')
            ).withColumn(
                'income_expense_ratio',
                F.col('monthly_income') / F.col('monthly_expenses')
            )

            return behavior_metrics.toPandas()

        except Exception as e:
            logger.error(f"Error in customer behavior analysis: {str(e)}")
            raise

    def generate_correlation_matrix(self) -> pd.DataFrame:
        """Generate correlation matrix for numerical variables"""
        logger.info("Generating correlation matrix")
        try:
            return self.df.select(self.numerical_cols).toPandas().corr()
        except Exception as e:
            logger.error(f"Error in correlation matrix generation: {str(e)}")
            raise

    def perform_complete_eda(self) -> Dict[str, Any]:
        """Perform complete EDA"""
        logger.info("Starting complete EDA process")
        try:
            results = {
                'data_quality': self.analyze_data_quality(),
                'categorical_analysis': self.analyze_categorical_variables(),
                'numerical_analysis': self.analyze_numerical_variables(),
                'risk_patterns': self.analyze_risk_patterns(),
                'customer_behavior': self.analyze_customer_behavior(),
                'correlation_matrix': self.generate_correlation_matrix()
            }

            logger.info("EDA process completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in complete EDA process: {str(e)}")
            raise

    def return_unique_clients_df(self) -> pd.DataFrame:
        """Return a DataFrame with unique clients"""
    # Columnas that define a unique client
        columns_unique = [
            'cli',
            'tipo_cli',
            'estado_cli',
            'nivel_academico',
            'profn',
            'ocup',
            'tipo_contrato',
            'genero_cli',
            'tipo_vivienda',
            'pais_nacim',
            'ing_mes',
            'egresos_mes',
            'origen_fondos',
            'operac_moneda_extranjera',
            'ciiu',
            'nivel_riesgo_ciiu',
            'sociedad_ccial_civ',
            'mun_res',
            'score_riesgo_mun'
        ]
        # Select unique clients based on the columns_unique
        unique_clients_df = self.df.select(columns_unique).distinct()

        # Get behavior metrics as Pandas DataFrame
        behavior_metrics_pd = self.analyze_customer_behavior()

        # Convert behavior metrics back to Spark DataFrame
        behavior_metrics_spark = self.spark.createDataFrame(behavior_metrics_pd)

        # Now join both Spark DataFrames
        result_spark = unique_clients_df.join(behavior_metrics_spark, on='cli', how='left')

        # Convert final result to Pandas as per return type annotation
        return result_spark

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save EDA results to disk"""
        logger.info(f"Saving EDA results to {output_path}")
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            logger.error(f"Error saving EDA results: {str(e)}")
            raise


    def load_results(self, input_path: str) -> Dict[str, Any]:
        """Load EDA results from disk"""
        logger.info(f"Loading EDA results from {input_path}")
        try:
            with open(input_path, 'rb') as f:
                results = pickle.load(f)
            return results
        except Exception as e:
            logger.error(f"Error loading EDA results: {str(e)}")
            raise


# Usage example:
"""
# Initialize and run EDA
spark_df = ... # your Spark DataFrame
eda = EDAAnalyzer(spark_df)
results = eda.perform_complete_eda()

# Access specific analyses
data_quality = results['data_quality']
risk_patterns = results['risk_patterns']
correlations = results['correlation_matrix']
"""