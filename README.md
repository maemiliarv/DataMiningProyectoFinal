Proyecto Final de Data Mining: Predicción de LTV, CAC y ROMI para Showz

📖 Contexto

En el PSet #4 se analizó métricas históricas de visitas, ventas y costos de Showz. Se construyeron modelos predictivos que estimen:

Lifetime Value (LTV) a 6–12 meses de cada cliente nuevo.

Customer Acquisition Cost (CAC) esperado por fuente y cohorte para el próximo trimestre.

Retorno sobre la inversión en marketing (ROMI) para cada canal de adquisición.

Se integraron técnicas de Machine Learning (regresión lineal, Ridge, Random Forest, XGBoost, LightGBM, CatBoost, ensambladores, validación con TimeSeriesSplit, explicabilidad con SHAP/PDP) y se generó una estrategia de asignación de presupuesto basada en simulaciones de ROMI.

🎯 Objetivos

Predecir el LTV de cada cliente nuevo en un horizonte de 6–12 meses.

Estimar el CAC por fuente y cohorte para el próximo trimestre.

Identificar los drivers clave de LTV y CAC (explicabilidad).

Proponer una estrategia de asignación de presupuesto utilizando predicciones de ROMI.

📂 Estructura del repositorio

├── data/
│   ├── raw/                # Datos originales (visits, orders, costs)
│   ├-─ processed/          # CSVs limpios para modeling
├── src/                    # Código fuente
│   ├── utils.py            # Funciones de carga, guardado y logging
│   ├── features.py         # Ingeniería de características
│   └── train.py            # Script de entrenamiento y evaluación
├── models/                 # Modelos entrenados (.pkl) y artefactos
├── notebooks/              # Notebooks de EDA, FE y resultados finales
    ├── 01_EDA.ipynb 
    ├── 02_FeatureEngineering.ipynb
    ├── 03_ModelTraining.ipynb 
    ├── 04_explicabilidad_Diagnostico.ipynb
    └── Final_Project_Showz_LTV_CAC.ipynb
├── requirements.txt        # Dependencias Python
└── README.md               # Documentación general

🛠️ Requisitos

Python 3.8+

pip

requirements.txt contiene:

pandas
numpy
scikit-learn
xgboost
lightgbm
catboost
shap
pyyaml
joblib

🚀 Instalación

# Clona este repositorio
git clone https://github.com/maemiliarv/DataMiningProyectoFinal.git
cd DataMiningProyectoFinal

# Crea y activa virtualenv (opcional)
python -m venv venv
source venv/bin/activate  # Linux/MacOSenv\Scripts\activate  # Windows

# Instala dependencias
pip install -r requirements.txt

⚙️ Configuración

Edita configs/config.yaml para ajustar rutas y parámetros:

data:
  input_path: data/processed/modeling_dataset.csv
  features:
    - revenue
    - costs
    - session_duration
    # ...otras features generadas
  target: LTV_180
  test_size: 0.2
  random_state: 42
model:
  output_path: models/LTV_180/final_model.pkl
  params:
    n_estimators: 100
    max_depth: 10
logging:
  file: logs/train.log

▶️ Ejecución

# Entrena el modelo de LTV
env/bin/python src/train.py --config configs/config.yaml

Los logs aparecerán en consola y en el archivo logs/train.log.

El modelo final se guardará en models/LTV_180/final_model.pkl.

Para entrenar CAC o probar otro target, duplica el bloque data.target y model.output_path en tu YAML.

📊 Resultados y Evaluación

Métricas: MAE, RMSE, MAPE sobre sets de validación temporal (TimeSeriesSplit) y test.

Explicabilidad: importancia de features (gain, permutation), SHAP/PDP para top 5.

Simulación de ROMI: usa predicciones de LTV y CAC para cuantificar retorno por canal ante distintos escenarios de presupuesto.

Revisa notebooks/Final_Project_Showz_LTV_CAC.ipynb para visualizaciones y análisis detallado.

🔍 Metodología

Este proyecto sigue el ciclo CRISP‑DM:

Business Understanding

Data Understanding

Data Preparation (ingeniería avanzada de features)

Modeling (baseline, avanzados, ensambladores)

Evaluation & Selection (TimeSeriesSplit + GridSearchCV)

Deployment & Simulation (estrategia de marketing)
