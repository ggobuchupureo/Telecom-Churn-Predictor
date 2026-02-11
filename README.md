# 📞 Telecom Churn Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-green.svg)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

Sistema de Machine Learning para predicción temprana de cancelación de clientes en empresas de telecomunicaciones, utilizando técnicas de ensamble y optimización de hiperparámetros.

---

## 📋 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Problema de Negocio](#-problema-de-negocio)
- [Dataset](#-dataset)
- [Metodología](#-metodología)
- [Pipeline de Procesamiento](#-pipeline-de-procesamiento)
- [Modelos Implementados](#-modelos-implementados)
- [Resultados](#-resultados)
- [Hallazgos Clave](#-hallazgos-clave)
- [Instalación y Uso](#-instalación-y-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías](#-tecnologías)
- [Conclusiones](#-conclusiones)
- [Contacto](#-contacto)

---

## Descripción del Proyecto

Este proyecto desarrolla un **modelo de ensamble** que permite predecir tempranamente si un cliente cancelará el servicio de telecomunicaciones (churn), además de identificar las características que más inciden en la separación de clientes. El objetivo es proporcionar a la empresa herramientas para implementar **estrategias de retención proactivas** y reducir la tasa de abandono.

---

## Problema de Negocio

La **cancelación de clientes** (churn) representa uno de los principales desafíos para las empresas de telecomunicaciones, impactando directamente en:

- **Ingresos recurrentes**: Pérdida de clientes activos y reducción del ARPU (Average Revenue Per User)
- **Costos de adquisición**: Adquirir un nuevo cliente cuesta 5-7 veces más que retener uno existente
- **Brand reputation**: Alta tasa de churn señaliza problemas de satisfacción del cliente

**Objetivo del modelo**: Identificar clientes con alto riesgo de abandono **antes** de que cancelen, permitiendo intervenciones personalizadas y cost-effective.

---

## Dataset

### Características del Dataset

- **Registros**: 3,333 clientes
- **Variables**: 11 atributos (10 predictores + 1 objetivo)
- **Distribución de clases**: 
  - Clase 0 (No Churn): 2,850 clientes (85.5%)
  - Clase 1 (Churn): 483 clientes (14.5%)
- **Valores nulos**: 0 (dataset limpio)

### Diccionario de Datos

| Variable | Tipo | Descripción |
|----------|------|-------------|
| **Churn** | int | **Variable objetivo**: 1 si el cliente canceló, 0 si no |
| AccountWeeks | int | Número de semanas con cuenta activa |
| ContractRenewal | int | 1 si renovó contrato recientemente, 0 si no |
| DataPlan | int | 1 si tiene plan de datos, 0 si no |
| DataUsage | float | GB de uso mensual de datos |
| CustServCalls | int | Número de llamadas al servicio al cliente |
| DayMins | float | Promedio de minutos diurnos al mes |
| DayCalls | int | Número medio de llamadas diurnas |
| MonthlyCharge | float | Factura mensual media ($) |
| OverageFee | float | Mayor cuota de exceso en últimos 12 meses ($) |
| RoamMins | float | Minutos de roaming |

---

## Metodología

El proyecto sigue el framework **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

### 1. **Business Understanding**
   - Definición del problema de churn
   - Identificación de métricas de éxito (Recall, Precision, F1-Score)
   - Establecimiento de objetivos: maximizar detección de clientes en riesgo

### 2. **Data Understanding**
   - Análisis exploratorio de datos (EDA)
   - Visualización de distribuciones
   - Análisis de correlaciones
   - Detección de outliers mediante boxplots

### 3. **Data Preparation**
   - Verificación de tipos de datos
   - División train/test (80/20)
   - Escalado de características (StandardScaler para SVM y Regresión Logística)
   - Balanceo de clases con SMOTE (Synthetic Minority Over-sampling Technique)

### 4. **Modeling**
   - Implementación de múltiples algoritmos
   - Optimización de hiperparámetros con GridSearchCV
   - Validación cruzada (5-fold CV)
   - Técnicas de ensamble (Bagging, Random Forest)

### 5. **Evaluation**
   - Comparación de modelos mediante métricas de clasificación
   - Análisis de feature importance
   - Identificación de clientes de alto riesgo

### 6. **Deployment**
   - Generación de predicciones probabilísticas
   - Ranking de clientes por probabilidad de churn
   - Recomendaciones para intervención

---

## Pipeline de Procesamiento

```python
# 1. Carga de datos
df = pd.read_csv('telecom_churn.csv')

# 2. División de datos
X = df.drop(columns=['Churn'])
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 3. Balanceo con SMOTE (solo para Bagging)
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# 4. Escalado (solo para SVM y Regresión Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenamiento del modelo
best_model = RandomForestClassifier(
    n_estimators=190,
    max_features='sqrt',
    oob_score=True
)
best_model.fit(X_train, y_train)

# 6. Predicción y evaluación
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]
```

---

## Modelos Implementados

### 1. **Decision Tree (Baseline)**
   - Modelo inicial sin optimización
   - **Problema detectado**: Overfitting severo (100% accuracy en train)
   - **Resultados Test**: Accuracy 87.9%, Recall 64.2%

### 2. **Decision Tree + GridSearchCV**
   - Optimización de hiperparámetros: `max_depth`, `min_samples_leaf`
   - Mejores parámetros: `max_depth=10`, `min_samples_leaf=0.01`
   - **Resultados Test**: Accuracy 92.4%, Precision 87.9%, Recall 53.7%
   - **Mejora**: Elimina overfitting, aumenta precisión

### 3. **Bagging Classifier + SMOTE**
   - 200 estimadores (árboles de decisión)
   - Balanceo de clases con SMOTE
   - **Resultados Test**: Accuracy 90.0%, Precision 61.5%, Recall 79.0%
   - **Trade-off**: Mayor recall a costa de menor precisión

### 4. **Bagging Heterogéneo**
   - Combinación de: Decision Tree, SVM (RBF), SVM (Sigmoid), Regresión Logística
   - **Resultados Test**: Accuracy 87.0%, Recall 16.0%
   - **Conclusión**: No supera al Bagging homogéneo (Decision Trees puros)

### 5. **Random Forest** ⭐
   - 45 estimadores, `class_weight='balanced'`
   - **Resultados Test**: Accuracy 93.2%, Precision 83.8%, Recall 65.3%
   - **Ventajas**: Balance óptimo entre métricas

### 6. **Random Forest + GridSearchCV** 🏆 **MEJOR MODELO**
   - Búsqueda exhaustiva de hiperparámetros
   - Mejores parámetros: `n_estimators=190`, `max_features='sqrt'`
   - **Resultados Test**: 
     - **Accuracy**: 93.7%
     - **Precision**: 83.5%
     - **Recall**: 69.5%
     - **F1-Score**: 75.9%
     - **ROC AUC**: 89.5%

---

## Resultados

### Comparativa de Modelos (Test Set)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|--------|----------|-----------|--------|----------|---------|
| Decision Tree (Baseline) | 87.9% | 56.5% | 64.2% | 60.1% | 78.0% |
| Decision Tree + GridSearch | 92.4% | 87.9% | 53.7% | 66.7% | 89.8% |
| Bagging + SMOTE | 90.0% | 61.5% | 79.0% | 69.1% | 90.2% |
| Bagging Heterogéneo | 87.0% | 62.0% | 16.0% | 25.0% | - |
| Random Forest | 93.2% | 83.8% | 65.3% | 73.4% | 89.9% |
| **Random Forest + GridSearch** | **93.7%** | **83.5%** | **69.5%** | **75.9%** | **89.5%** |

### Interpretación de Métricas (Modelo Final)

- **Accuracy 93.7%**: El modelo clasifica correctamente 93.7% de todos los clientes
- **Precision 83.5%**: De los clientes que el modelo predice que abandonarán, 83.5% realmente lo hacen (bajo falsos positivos)
- **Recall 69.5%**: El modelo detecta 69.5% de todos los clientes que realmente abandonan
- **F1-Score 75.9%**: Balance harmónico entre Precision y Recall
- **ROC AUC 89.5%**: Excelente capacidad discriminatoria entre clases

### Matriz de Confusión (Random Forest + GridSearch)

```
                 Predicho: No Churn    Predicho: Churn
Real: No Churn         560                 12
Real: Churn             29                 66
```

- **True Negatives (TN)**: 560 - Clientes fieles correctamente identificados
- **False Positives (FP)**: 12 - Clientes fieles erróneamente marcados como riesgo
- **False Negatives (FN)**: 29 - Clientes en riesgo **no detectados** (30.5% de los churns)
- **True Positives (TP)**: 66 - Clientes en riesgo correctamente identificados

---

## Hallazgos Clave

### 1. **Feature Importance (Random Forest + GridSearch)**

Las 4 variables más influyentes en la predicción de churn:

| Variable | Importancia | Insight de Negocio |
|----------|-------------|--------------------|
| **DayMins** | ~0.30 | Alto uso de minutos diurnos indica engagement, bajo uso señaliza riesgo |
| **CustServCalls** | ~0.23 | Múltiples llamadas a servicio al cliente correlacionan fuertemente con churn |
| **MonthlyCharge** | ~0.12 | Facturas elevadas sin valor percibido aumentan probabilidad de cancelación |
| **ContractRenewal** | ~0.09 | Falta de renovación reciente es señal temprana de desengagement |

**Recomendaciones**:
- Monitorear clientes con **3+ llamadas al servicio al cliente** → Intervención proactiva
- Segmentar clientes con **bajo uso de minutos** (<143.7 min/mes) → Ofertas de retención
- Revisar percepción de valor en clientes con **MonthlyCharge > $66** → Programas de fidelización

### 2. **Trade-offs entre Modelos**

- **Decision Tree + GridSearch**: 
  - ✅ Alta precisión (87.9%) → Pocos falsos positivos
  - ❌ Bajo recall (53.7%) → Pierde muchos clientes en riesgo
  
- **Bagging + SMOTE**: 
  - ✅ Alto recall (79.0%) → Detecta más churns
  - ❌ Baja precisión (61.5%) → Genera muchos falsos positivos (desperdicio de recursos en retención)
  
- **Random Forest + GridSearch** (seleccionado): 
  - ✅ Balance óptimo (Precision 83.5%, Recall 69.5%)
  - ✅ Mejor ROC AUC (89.5%)
  - **Justificación**: Maximiza valor de negocio minimizando costos de campaña (baja FP) y pérdida de clientes (aceptable FN)

---

## Tecnologías

### Lenguaje
- **Python 3.8+**

### Librerías de Data Science
- **Pandas 1.3+**: Manipulación y análisis de datos
- **NumPy 1.21+**: Operaciones numéricas
- **Matplotlib 3.4+**: Visualización estática
- **Seaborn 0.11+**: Visualización estadística avanzada

### Machine Learning
- **Scikit-learn 1.0+**: 
  - Modelos: `DecisionTreeClassifier`, `RandomForestClassifier`, `BaggingClassifier`, `SVC`, `LogisticRegression`
  - Preprocessing: `StandardScaler`, `train_test_split`
  - Optimización: `GridSearchCV`, `RandomizedSearchCV`
  - Métricas: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`
- **Imbalanced-learn 0.8+**: `SMOTE` (balanceo de clases)
- **Statsmodels 0.13+**: Análisis estadístico

### Entorno
- **Jupyter Notebook**: Desarrollo interactivo
- **Git**: Control de versiones

---

## Conclusiones

### Fortalezas del Modelo

**Alta precisión general (93.7%)**: El modelo identifica correctamente la mayoría de los clientes

**Excelente precisión en clase positiva (83.5%)**: De los clientes marcados como riesgo, 83.5% realmente lo son → Bajo desperdicio de recursos en campañas de retención

**Recall aceptable (69.5%)**: Detecta 7 de cada 10 clientes que abandonarán → Permite intervenciones preventivas efectivas

**Feature importance interpretable**: Las variables más importantes (`DayMins`, `CustServCalls`) tienen sentido de negocio claro y son accionables

**ROC AUC alto (89.5%)**: Excelente capacidad discriminatoria entre clientes fieles y en riesgo

### Limitaciones

**Recall no perfecto (69.5%)**: El modelo no detecta ~30% de los churns reales → Algunos clientes en riesgo no reciben intervención

**Desbalance de clases**: Solo 14.5% de la muestra es churn → El modelo podría estar sesgado hacia la clase mayoritaria

**Datos estáticos**: El dataset no captura comportamiento temporal ni tendencias → El modelo no puede predecir cambios bruscos en engagement

**Variables limitadas**: No incluye información sobre:
   - Historial de pagos / morosidad
   - Satisfacción del cliente (NPS)
   - Competencia / ofertas externas
   - Interacciones digitales (app, web)

### Trabajo Futuro

**Mejoras Técnicas**:
- Implementar **modelos de boosting** (XGBoost, LightGBM, CatBoost) para mejorar recall
- Explorar **redes neuronales** para capturar relaciones no lineales complejas
- Aplicar **técnicas de ensemble stacking** combinando múltiples modelos
- Incorporar **validación temporal** (time-based split) si se obtienen datos históricos

**Mejoras de Datos**:
- Agregar **variables de comportamiento temporal** (tendencias de uso, decaimiento de engagement)
- Incluir **datos de interacciones omnichannel** (email, chat, app)
- Incorporar **variables de satisfacción** (NPS, CSAT scores)
- Enriquecer con **datos externos** (ofertas de competidores, condiciones económicas)

**Despliegue en Producción**:
- Desarrollar **API REST** para integración con CRM
- Implementar **re-entrenamiento automático** con nuevos datos
- Crear **dashboard interactivo** para visualización de riesgo de churn por segmento
- Establecer **sistema de alertas** para clientes de alto riesgo

---

## Contacto

**Gastón González Ovalle**  
Data Scientist | Bioingeniería + Machine Learning

- Email: [ggobuchupureo@gmail.com](mailto:ggobuchupureo@gmail.com)
- LinkedIn: [linkedin.com/in/gaston-gonzalez-ovalle](https://www.linkedin.com/in/gaston-gonzalez-ovalle/)
- GitHub: [github.com/ggobuchupureo](https://github.com/ggobuchupureo)

---
