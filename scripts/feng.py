from pyspark.sql import functions as F


# Creacion de variables dummies según EDA
CATEGORICAL_DUMMIES = {
    "tipo_cli": ["persona natural"],
    "estado_cli": ["activo", "inactivo"],
    "nivel_academico": ["None", "universitario", "bachiller", "no informa", "tecnologo", "especializacion"],
    "tipo_contrato": ["None", "termino indefinido", "no informa/no tiene"],
    "genero_cli": ["m", "f"],
    "tipo_vivienda": ["None", "familiar", "propia", "alquilada", "no informa"],
    "pais_nacim": ["colombia"],
    "operac_moneda_extranjera": ["n"],
    "nivel_riesgo_ciiu": ["None", "bajo", "medio", "alto"],
    "trn_desc_tip_cta": ["ahorro", "corriente"],
    "trn_efec": ["no", "None", "si"],
    "trn_canal_serv_efec": ["sin informacion", "None", "corresponsal bancario", "cajero"],
    "mun_res": ["bogota d.c.", "medellin", "cali", "Other", "None"],
    'trn_oper': ['credit', 'debit']
}

OCCUPATION_CATEGORIES = {
    "Professionals": ["profesional independiente", "socio o empleado - socio"],
    "Self-employed": ["independiente", "comerciante", "rentista de capital", "agricultor", "ganadero"],
    "Non-working": ["pensionado", "ama de casa", "estudiante", "desempleado con ingresos", "desempleado sin ingresos"],
    "Others": ["None", "otra"]
}

PROFESSION_CATEGORIES = {
    "STEM": [
        "ingenieria de sistemas", "ingenieria industrial", "ingenieria civil",
        "ingenieria mecanica", "ingeniero electronico", "ingenieria electrica",
        "tecnologia sistemas", "ingeniria quimica", "tecnologia electricidad",
        "tecnologia mecanica", "tecnologia industrial", "ingenieria ambiental",
        "ingenieria administrativa", "biologia", "ingenieria de petroleos",
        "tecnologia agropecuaria", "ingenieria agricola", "ingenieria financiera",
        "tecnologia en construccion", "geologia", "ingenieria forestal",
        "ingenieria de minas", "ingenieria sanitaria", "ingeniero metalurgico",
        "tecnologia en minas", "arquitectura", "agronomia"
    ],

    "Health & Medicine": [
        "medicina", "enfermeria", "odontologia", "quimica farmaceutica",
        "nutricion y dietetica", "auxiliar de enfermeria", "regencia de farmacia",
        "auxiliar de odontologia", "operaciones de equipos medicos",
        "tecnologia en ciencias de la salud", "veterinaria", "bacteriologia"
    ],

    "Business, Law & Administration": [
        "administracion", "contaduria", "derecho", "economia",
        "auxiliar contable", "mercadotecnia", "comercio internacional",
        "secretariado", "tecnologia en administracion", "transportador",
        "carrera militar", "pilotos", "azafata"
    ],

    "Arts, Humanities & Social Sciences": [
        "educacion", "psicologia", "comunicacion social", "diseño y publicidad",
        "artes", "trabajo social", "profesores de educacion primaria",
        "profesores de educacion preescolar", "sociologia", "filosofia y letras",
        "deportistas entrenadores tecnicos deport", "musicos artistas empresarios y prud espect",
        "escritores periodistas y trabajadores simil", "fotografos y operadores de camara cine y tv",
        "escultores pintores fotografos y art simi", "sacerdote", "religiosa"
    ]
}

def feature_engineering(df, df_clients):
    """Feature engineering for the dataset"""
    columnas_transacciones = [
        'trn_desc_tip_cta',
        'trn_oper',
        'trn_efec',
        'trn_canal_serv_efec'
    ]
    df_copy = df.select(["cli"] + columnas_transacciones)
    df_clients_copy = df_clients
    # Create dummies for categorical variables
    for col, categories in CATEGORICAL_DUMMIES.items():
        if col not in columnas_transacciones:
            for cat in categories:
                df_clients_copy = df_clients_copy.withColumn(f"{col}_{cat}", F.when(F.col(col) == cat, 1).otherwise(0))
        elif col in columnas_transacciones:
            for cat in categories:
                df_copy = df_copy.withColumn(f"{col}_{cat}", F.when(F.col(col) == cat, 1).otherwise(0))
        else:
            raise ValueError(f"Column {col} not found in the data")

    df_clients_copy = df_clients_copy.withColumn("ocup_category", F.when(F.col("ocup").isin(OCCUPATION_CATEGORIES["Professionals"]), "Professionals")
                                                 .when(F.col("ocup").isin(OCCUPATION_CATEGORIES["Self-employed"]), "Self-employed")
                                                 .when(F.col("ocup").isin(OCCUPATION_CATEGORIES["Non-working"]), "Non-working")
                                                 .when(F.col("ocup").isin(OCCUPATION_CATEGORIES["Others"]), "Others")
                                                 .otherwise("Others"))


    df_clients_copy = df_clients_copy.withColumn("profn_category", F.when(F.col("profn").isin(PROFESSION_CATEGORIES["STEM"]), "STEM")
                                                 .when(F.col("profn").isin(PROFESSION_CATEGORIES["Health & Medicine"]), "Health & Medicine")
                                                 .when(F.col("profn").isin(PROFESSION_CATEGORIES["Business, Law & Administration"]),
                                                       "Business, Law & Administration")
                                                 .when(F.col("profn").isin(PROFESSION_CATEGORIES["Arts, Humanities & Social Sciences"]),
                                                       "Arts, Humanities & Social Sciences")
                                                 .otherwise("Others"))
    # create dummies for occupation and profession categories
    for col in ["ocup_category", "profn_category"]:
        for cat in df_clients_copy.select(col).distinct().rdd.flatMap(lambda x: x).collect():
            df_clients_copy = df_clients_copy.withColumn(f"{col}_{cat}", F.when(F.col(col) == cat, 1).otherwise(0))

    # Drop original columns and CATEGORICAL_DUMMIES.keys() plus profn_category and ocup_category
    df_clients_copy = df_clients_copy.drop(*CATEGORICAL_DUMMIES.keys())
    df_clients_copy = df_clients_copy.drop(*["profn", "ocup", "sociedad_ccial_civ", "ciiu", "origen_fondos"])
    df_clients_copy = df_clients_copy.drop(*["profn_category", "ocup_category"])

    df_copy = df_copy.drop(*columnas_transacciones )
    # create a data frame aggregating median the number of transactions per client in all columns of df_copy
    df_agg_median = df_copy.groupBy("cli").agg(*[F.median(col).alias(f"{col}_median") for col in df_copy.columns[1:]])
    # join df_agg_median with df_clients_copy
    df_clients_copy = df_clients_copy.join(df_agg_median, on="cli", how="left")

    return df_clients_copy


def save_clients_df(df, output_dir):
    """Save the clients dataframe to disk"""
    df.write.mode("overwrite").parquet(output_dir)
    return None