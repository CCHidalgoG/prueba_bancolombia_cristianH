import logging
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from typing import List, Any
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler,  StringIndexer
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, when, coalesce, lit

# Iniciar Spark Session
spark = SparkSession.builder \
    .appName("ClientSegmentation") \
    .getOrCreate()

# Cargar datos parquet
df = spark.read.parquet('output/clients_df')

def validar_columnas(df, numeric_cols, categorical_cols):
    # Obtener todas las columnas del DataFrame
    columnas_df = set(df.columns)
    # Convertir las listas de columnas a sets para la comparación
    numeric_set = set(numeric_cols)
    categorical_set = set(categorical_cols)

    # Verificar columnas numéricas faltantes
    numeric_faltantes = numeric_set - columnas_df

    # Verificar columnas categóricas faltantes
    categorical_faltantes = categorical_set - columnas_df

    # Imprimir resultados
    if len(numeric_faltantes) > 0:
        print("Columnas numéricas faltantes:", sorted(list(numeric_faltantes)))

    if len(categorical_faltantes) > 0:
        print("Columnas categóricas faltantes:", sorted(list(categorical_faltantes)))

    # Retornar True si todas las columnas están presentes
    return len(numeric_faltantes) == 0 and len(categorical_faltantes) == 0


def limpiar_nombres_columnas(columnas):
    columnas_limpias = []
    for col in columnas:
        # Reemplazar espacios, puntos, y caracteres especiales
        nuevo_nombre = (col.lower()
                        .replace(' ', '_')
                        .replace('.', '')
                        .replace('&', 'and')
                        .replace('/', '_')
                        .replace('-', '_')
                        .replace(',', ''))
        columnas_limpias.append(nuevo_nombre)
    return columnas_limpias


def prep_data_inic(df):
    # crear df_limpio cambiando los nombres de las columnas
    df_limpio = df.toDF(*limpiar_nombres_columnas(df.columns))


    # variables numéricas
    numeric_cols = ['ing_mes', 'egresos_mes', 'score_riesgo_mun', 'avg_transaction_amount', 'transaction_count',
                    'cash_transactions', 'monthly_income', 'monthly_expenses', 'income_expense_ratio']
    categorical_cols = [
        'tipo_cli_persona_natural', 'estado_cli_activo', 'estado_cli_inactivo',
        'nivel_academico_none', 'nivel_academico_universitario', 'nivel_academico_bachiller',
        'nivel_academico_no_informa', 'nivel_academico_tecnologo', 'nivel_academico_especializacion',
        'tipo_contrato_none', 'tipo_contrato_termino_indefinido', 'tipo_contrato_no_informa_no_tiene',
        'genero_cli_m', 'genero_cli_f', 'tipo_vivienda_none', 'tipo_vivienda_familiar',
        'tipo_vivienda_propia', 'tipo_vivienda_alquilada', 'tipo_vivienda_no_informa',
        'pais_nacim_colombia', 'operac_moneda_extranjera_n', 'nivel_riesgo_ciiu_none',
        'nivel_riesgo_ciiu_bajo', 'nivel_riesgo_ciiu_medio', 'nivel_riesgo_ciiu_alto',
        'mun_res_bogota_dc', 'mun_res_medellin', 'mun_res_cali', 'mun_res_other', 'mun_res_none',
        'ocup_category_self_employed', 'ocup_category_others', 'ocup_category_non_working',
        'ocup_category_professionals', 'profn_category_health_and_medicine', 'profn_category_stem',
        'profn_category_others', 'profn_category_arts_humanities_and_social_sciences',
        'profn_category_business_law_and_administration', 'trn_desc_tip_cta_ahorro_median',
        'trn_desc_tip_cta_corriente_median', 'trn_efec_no_median', 'trn_efec_none_median',
        'trn_efec_si_median', 'trn_canal_serv_efec_sin_informacion_median',
        'trn_canal_serv_efec_none_median', 'trn_canal_serv_efec_corresponsal_bancario_median',
        'trn_canal_serv_efec_cajero_median', 'trn_oper_credit_median', 'trn_oper_debit_median'
    ]

    print(validar_columnas(df_limpio, numeric_cols, categorical_cols))


    df_limpio = df_limpio.withColumn(
        'score_riesgo_mun',
        when(col('score_riesgo_mun').isNull(), 0.0)  # Manejar nulos
        .otherwise(col('score_riesgo_mun').cast('double'))  # Convertir a double
    )

    return df_limpio, numeric_cols, categorical_cols


df_limpio, numeric_cols, categorical_cols = prep_data_inic(df)


def crear_pipeline_segmentacion(numeric_cols, categorical_cols):
    """Crea un pipeline de transformación para la segmentación"""
    # Vector Assembler para todas las características
    assembler = VectorAssembler(
        inputCols=numeric_cols + categorical_cols,
        outputCol="numeric_features",
        handleInvalid="skip"
    )

    # Standard Scaler solo para el vector completo
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    return Pipeline(stages=[assembler, scaler])


def evaluar_kmeans(data_clust, max_clusters=20, min_prop=0.01, max_prop=0.45):
    """Evalúa diferentes números de clusters y retorna métricas"""
    silhouette_scores = []
    proportions = []
    valid_k = []
    evaluator = ClusteringEvaluator(
        predictionCol='prediction',
        featuresCol='features',
        metricName='silhouette'
    )

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(k=k, featuresCol='features', seed=42)
        model = kmeans.fit(data_clust)
        predictions = model.transform(data_clust)

        # Calcular silhouette score
        score = evaluator.evaluate(predictions)

        # Calcular proporciones
        cluster_counts = predictions.groupBy('prediction').count().toPandas()
        total = cluster_counts['count'].sum()
        props = cluster_counts['count'] / total

        # Verificar si las proporciones están en el rango deseado
        if props.min() >= min_prop and props.max() <= max_prop:
            silhouette_scores.append(score)
            proportions.append((props.min(), props.max()))
            valid_k.append(k)

    return silhouette_scores, proportions, valid_k


def crear_modelo_segmentacion(clean_df, numeric_cols_, categorical_cols_):
    """
    Crea y entrena el modelo de segmentación

    Returns:
        tuple: (predicciones_finales, mejor_k, mejor_puntuacion_silhouette, modelo_final)
    """

    try:
        for col_name in numeric_cols_:
            clean_df = clean_df.withColumn(col_name, coalesce(col(col_name), lit(0.0)))
    except:
        pass

    # Crear y ejecutar pipeline simplificado
    pipeline = crear_pipeline_segmentacion(numeric_cols_, categorical_cols_)
    pipeline_model = pipeline.fit(clean_df)
    dataset = pipeline_model.transform(clean_df)

    # Evaluar diferentes números de clusters
    silhouette_scores, proportions, valid_k = evaluar_kmeans(dataset)

    if not valid_k:
        raise ValueError("No se encontraron clusters que cumplan con las proporciones requeridas")

    # Encontrar mejor número de clusters
    best_k_idx = max(range(len(silhouette_scores)), key=lambda i: silhouette_scores[i])
    best_k = valid_k[best_k_idx]

    # Crear modelo final
    final_kmeans = KMeans(k=best_k, featuresCol='features', seed=42)
    final_model = final_kmeans.fit(dataset)

    # Agregar predicciones y características importantes
    final_predictions = final_model.transform(dataset)

    return final_predictions, best_k, silhouette_scores[best_k_idx], silhouette_scores, valid_k, final_model


def visualizar_resultados(predictions, silhouette_scores, valid_k, path):
    """Visualiza los resultados de la segmentación"""
    # Convertir a pandas para visualización
    cluster_props = predictions.groupBy('prediction').count().toPandas()
    cluster_props['proportion'] = cluster_props['count'] / cluster_props['count'].sum()

    plt.figure(figsize=(12, 5))

    # Gráfico de silhouette scores
    plt.subplot(1, 2, 1)
    plt.plot(valid_k, silhouette_scores, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score por número de clusters')

    # Gráfico de proporciones
    plt.subplot(1, 2, 2)
    sns.barplot(data=cluster_props, x='prediction', y='proportion')
    plt.xlabel('Cluster')
    plt.ylabel('Proporción')
    plt.title('Proporción de clientes por cluster')

    plt.tight_layout()
    plt.show()
    plt.savefig(path)

    return cluster_props


def select_k_features_random_forest(
        df,
        target_col: str,
        numcols: List[str],
        catcols: List[str]) -> tuple[RandomForestClassificationModel, list[tuple[str, Any]]]:
    """
    Selecciona las k características más importantes usando Random Forest.

    Args:
        df: DataFrame de Spark
        target_col: Nombre de la columna objetivo
        numcols: Lista de columnas numéricas
        catcols: Lista de columnas categóricas
        k: Número de características a seleccionar (default: 15)

    Returns:
        Tuple con:
        - Lista de columnas numéricas seleccionadas
        - Lista de columnas categóricas seleccionadas
        - Modelo Random Forest entrenado

    Raises:
        ValueError: Si k es mayor que el número total de características
        Exception: Para otros errores durante el proceso
    """
    try:

        logging.info("Iniciando proceso de selección de características")

        # Convertir columnas categóricas a índices
        indexers = [
            StringIndexer(
                inputCol=col,
                outputCol=f"{col}_idx",
                handleInvalid='keep'
            )
            for col in catcols
            # Not in targe_col and cli
            if 'idx' not in col and col != target_col and col != 'cli'
        ]

        if not indexers:
            logging.warning("No se encontraron columnas categóricas para indexar")

        # Crear y ajustar pipeline de indexación
        pipeline = Pipeline(stages=indexers)
        cols_train = [target_col] + numcols + catcols
        df_indexed = pipeline.fit(df.select(*cols_train)).transform(df.select(*cols_train))

        # Preparar features para el modelo
        feature_cols = [col for col in df_indexed.columns if 'idx' in col]
        if not feature_cols:
            raise ValueError("No se encontraron características después de la indexación")

        # Vector Assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol='features',
            handleInvalid='keep'
        )
        df_assembled = assembler.transform(df_indexed)

        # Configurar y entrenar Random Forest
        rf = RandomForestClassifier(
            featuresCol='features',
            labelCol=target_col,
            numTrees=100,
            seed=42
        )
        model = rf.fit(df_assembled)

        # Calcular importancia de características
        importances = model.featureImportances.toArray()
        feature_names = df_assembled.columns
        feature_importances = list(zip(feature_names, importances))

        # Seleccionar top k características
        top_k_features = sorted(
            feature_importances,
            key=lambda x: x[1],
            reverse=True
        )

        return model, top_k_features

    except Exception as e:
        logging.error(f"Error en la selección de características: {str(e)}")
        raise




def main():
    # Ejecutar el proceso de segmentación
    try:
        print(f"El número de features iniciales es de: {len(numeric_cols + categorical_cols)}")

        try:
            # verificar si el modelo está guardado
            modelo = KMeans.load('output/modelo0')
            predictions = spark.read.parquet('output/predictions0')
        except:
            # Crear y entrenar el modelo
            predictions, best_k, best_score, siluetas, ks, modelo = crear_modelo_segmentacion(df_limpio, numeric_cols, categorical_cols)
            print("Para el modelo inicial, el mejor número de clusters es:", best_k)
            print("El mejor silhouette score es:", best_score)
            # guardar modelo como modelo0
            modelo.write().overwrite().save('output/modelo0')
            # save predictions

            predictions.write.mode("overwrite").parquet('output/predictions0')
        # Seleccionar las k características más importantes
        if "prediction" in predictions.columns:
            dataset = predictions.withColumnRenamed("prediction", "cluster")

        # Tratar de cargar el modelo rfmodel
        try:
            rfmodel = RandomForestClassificationModel.load('output/rfmodel')
            print("Modelo Random Forest cargado")
            # load top_k_feat
            with open('output/top_k_feat.txt', 'r') as f:
                top_k_feat = [tuple(line.strip().split(',')) for line in f]
            print("top_k_feat cargado")
        except:
            rfmodel, top_k_feat = select_k_features_random_forest(dataset.drop(*['features', 'numeric_features']),
                                                                  'cluster',
                                                                  numeric_cols,
                                                                  categorical_cols)
            # guardar rfmodel
            rfmodel.write().overwrite().save('output/rfmodel')
            # guardar top_k_feat, overwriting
            with open('output/top_k_feat.txt', 'w') as f:
                for col, importance in top_k_feat:
                    f.write(f"{col},{importance}\n")



        # inner del top k vs numeric_cols and categorical_cols
        top_k_feat_ = [(col, importance) for col, importance in top_k_feat if col != 'cluster']
        # top_k_feat
        for i, (col, importance) in enumerate(top_k_feat_):
            print(f"{i + 1}. {col}: {float(importance):.4f}")

        K = 7
        # filtrar cluster de top_k_feat
        categorical_cols_reduce = [col for col, _ in top_k_feat_[:K] if col in categorical_cols]
        numeric_cols_reduce = [col for col, _ in top_k_feat_[:K] if col in numeric_cols]


        # reentrenar el modelo con las características seleccionadas
        predictions_, best_k_, best_score_, siluetas_, ks_, modelo_ = crear_modelo_segmentacion(df_limpio, numeric_cols_reduce, categorical_cols_reduce)
        print("Para el modelo optimizado, el mejor número de clusters es:", best_k_)
        print("El mejor silhouette score es:", best_score_)
        # guardar modelo como modelo
        modelo.write().overwrite().save('output/modelo')

        cluster_props = visualizar_resultados(predictions_, ks_, siluetas_, 'resultados/cluster_props.png')

        # resumir los resultados haciendo un groupby por cluster y mediana de todas las columnas
        resumen_cluster = predictions_.groupBy('prediction').agg(
            *[F.round(F.median(col), 2).alias(f"median_{col}") for col in numeric_cols_reduce+categorical_cols_reduce]
        )
        # realizar un gráfico de coordenas paralelas con resumen_cluster con plotly

        fig = px.parallel_coordinates(predictions_.drop(*['cli','features', 'numeric_features']).toPandas().sample(10_000), color='prediction')
        # rotate feature names
        fig.update_layout(
            xaxis=dict(tickangle=90),
            title="Parallel Coordinates Plot"
        )
        fig.write_html('resultados/parallel_coordinates.html')

        # save predicitions_
        predictions_.write.mode("overwrite").parquet('output/predictions_final.parquet')
    except:
        print("Error en el proceso de segmentación")
        spark.stop()
        raise



if __name__ == '__main__':
    main()
