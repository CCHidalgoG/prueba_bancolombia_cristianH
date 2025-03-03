# Proyecto de Análisis de Datos en Python

## Descripción
Este proyecto se centra en la exploración y análisis de datos utilizando Python 3.11.10. Con un énfasis en datos estructurados, se emplean herramientas y bibliotecas para una manipulación y visualización eficientes. Además, se utiliza **Poetry** para la gestión de dependencias y el entorno del proyecto.

## Requisitos del Sistema
- **Python**: 3.11.10
- **Poetry**: Herramienta de gestión de dependencias. [Cómo instalar Poetry](https://python-poetry.org/docs/#installation)
- **Sistema Operativo**: Linux (amd64)
- **Entorno de Desarrollo**: DataSpell 2024.3.1.1 (opcional)

## Dependencias
Las bibliotecas utilizadas en este proyecto incluyen:

- **matplotlib**: Visualización de gráficos.
- **numpy**: Cálculos matemáticos rápidos.
- **pandas**: Manipulación y análisis de datos tabulares.
- **pillow**: Procesamiento y manipulación de imágenes.
- **plotly**: Visualización interactiva de datos.
- **pyparsing**: Procesamiento de texto.
- **pyspark**: Procesamiento distribuido.
- **pytz**: Manejo de zonas horarias.
- **requests**: Interacciones HTTP.
- **seaborn**: Extensiones de visualización para Matplotlib.
- **six**: Compatibilidad entre Python 2 y 3.

Esquema del proyecto
```
proyecto-analisis-datos/
├── .venv/
├── data/
├── output/
├── resultados/
├── scripts/
├── pyproject.toml
├── doc_varios.md
├── .gitignore
└── README.md
```

Estas dependencias están definidas en el archivo `pyproject.toml` de **Poetry**.

Estos son los pasos recomendados para instalar y configurar el entorno del proyecto:

1. **Clonar el repositorio del proyecto**:
   Primero obten el código en tu máquina local:
   ```bash
   git clone https://github.com/usuario/proyecto-analisis-datos.git
   cd proyecto-analisis-datos
   ```

2. **Instalar Poetry**:
   Sigue las instrucciones provistas en la [documentación oficial de Poetry](https://python-poetry.org/docs/#installation).

3. **Instalar las dependencias del proyecto**:
   Desde la raíz del repositorio clonado, ejecuta:
   ```bash
   poetry install
   ```

4. **Activar el entorno virtual**:
   Una vez instales las dependencias, activa el entorno virtual para empezar a usar los scripts:
   ```bash
   poetry shell
   ```

## Uso del Proyecto

El proyecto provee scripts principales que se encuentran en la carpeta `scripts/` para realizar las labores de análisis de datos. Algunos pasos generales:

- Ejecuta el script principal:
  ```bash
  python scripts/nombre_del_script.py
  ```

- Todos los resultados generados se almacenarán en la carpeta `resultados/` o `output/`, dependiendo del tipo de exportación.

## Colaboración

Si deseas colaborar en este proyecto, asegúrate de que tus cambios pasen las siguientes comprobaciones:

1. Haz un fork del proyecto.
2. Trabaja en una rama nueva:
   ```bash
   git checkout -b nombre-de-tu-rama
   ```
3. Realiza tus cambios y escribe un mensaje claro en el commit.
4. Envía una Pull Request describiendo tus cambios!


