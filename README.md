# Proyecto Final de Data Mining: Predicción de LTV, CAC y ROMI para Showz

---

## Contexto

En el PSet #4 se analizó métricas históricas de visitas, ventas y costos de Showz. Se construyeron modelos predictivos que estimen:

- **Lifetime Value (LTV)** a 6–12 meses por cliente nuevo.
- **Customer Acquisition Cost (CAC)** esperado por fuente y cohorte para el próximo trimestre.
- **Retorno sobre la Inversión en Marketing (ROMI)** para cada canal de adquisición.

Se integraron técnicas de Machine Learning (regresión lineal, Ridge, Random Forest, XGBoost, LightGBM, CatBoost, ensambladores, validación con TimeSeriesSplit, explicabilidad con SHAP/PDP) y se generó una estrategia de asignación de presupuesto basada en simulaciones de ROMI.

---

## Objetivos

- Predecir el **LTV** de cada cliente nuevo en un horizonte de 6–12 meses.
- Estimar el **CAC** por fuente y cohorte para el próximo trimestre.
- Identificar los **drivers clave** de LTV y CAC mediante análisis de explicabilidad.
- Proponer una estrategia de marketing basada en predicciones de **ROMI** por canal.

## Estructura del repositorio

```bash
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
```

---

## Requisitos

- Python 3.8 o superior
- pip
- Paquetes necesarios (definidos en `requirements.txt`):

  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost
  - shap
  - pyyaml
  - joblib  

---

## Instalación

1. **Clona este repositorio**
```bash
git clone https://github.com/maemiliarv/DataMiningProyectoFinal.git
cd DataMiningProyectoFinal
```

2. **Crea y activa virtualenv (opcional)**
```bash
python -m venv venv
source venv/bin/activate      # En Linux/Mac
venv\Scripts\activate         # En Windows
```

3. **Instala dependencias**
```bash
pip install -r requirements.txt
```

---

## Configuración
Edita configs/config.yaml para ajustar rutas y parámetros:

```bash
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
```

Para entrenar CAC o probar otro target, duplica el bloque data.target y model.output_path en tu YAML.

---


## Ejecución

1. **Entrena el modelo de LTV**
```bash
python src/train.py --config configs/config.yaml
```

2. **El log de ejecución se guarda en:**
```bash
logs/train.log
```

3. **El modelo entrenado se guarda como:**
```bash
models/LTV_180/final_model.pkl.
```

---

## Resultados y Evaluación

- **Métricas evaluadas:**
MAE, RMSE, MAPE sobre sets de validación temporal (TimeSeriesSplit) y test.

- **Explicabilidad:**
Importancia de variables (gain), gráficas PDP para las top 5.

- **Simulación de ROMI:**
Uso de predicciones de LTV y CAC para cuantificar retorno por canal ante distintos escenarios de presupuesto.

Consulta *notebooks/Final_Project_Showz_LTV_CAC.ipynb* para visualizaciones y análisis detallado.

---


## Metodología

El proyecto sigue el ciclo **CRISP‑DM**:

1. Business Understanding.
2. Data Understanding.
3. Data Preparation (ingeniería avanzada de features).
4. Modeling (baseline, avanzados, ensambladores).
5. Evaluation & Selection (TimeSeriesSplit + GridSearchCV).
6. Deployment & Simulation (estrategia de marketing)

---
