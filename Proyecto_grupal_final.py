"""
Análisis Completo de Churn (Abandono de Clientes) en Telecomunicaciones
Integración de análisis exploratorio, inferencial y modelado predictivo
Dataset: Telco_customer_churn.xlsx
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo global, sólo una vez
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 7)
sns.set_palette("husl")

# --- Carga y preparación de datos ---

def cargar_datos(ruta_excel='Telco_customer_churn.xlsx'):
    try:
        df = pd.read_excel(ruta_excel, sheet_name='Telco_customer_churn')
        print("Datos cargados correctamente.")

        # Limpieza: convertir 'Total Charges' a numérico y rellenar NA con 0
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'] = df['Total Charges'].fillna(0)

        # Crear columna numérica para churn
        df['Churn Numeric'] = df['Etiqueta Churn'].apply(lambda x: 1 if x == 'Si' else 0)

        # Mapear Senior Citizen a categórico
        df['Senior Citizen'] = df['Senior Citizen'].map({0: 'No', 1: 'Si'})

        return df
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return None

# --- Análisis Exploratorio Básico (de Avance proyecto) ---

def analisis_exploratorio_basico(df):
    print("\nResumen estadístico de columnas numéricas:")
    print(df.describe())
    print("\nDistribución de la variable objetivo (Etiqueta Churn):")
    print(df['Etiqueta Churn'].value_counts())

    # Gráficos exploratorios básicos
    # 1. Distribución de Churn
    plt.figure(figsize=(8, 6))
    df['Etiqueta Churn'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribución de Churn (Baja de clientes)')
    plt.xlabel('Estado del Cliente')
    plt.ylabel('Cantidad de Clientes')
    plt.xticks(rotation=0)
    plt.savefig('churn_distribution.png')
    plt.show()

    # 2. Churn por género
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Etiqueta Churn', data=df)
    plt.title('Churn (Baja de clientes) por Género')
    plt.xlabel('Género')
    plt.ylabel('Cantidad')
    plt.savefig('churn_by_gender.png')
    plt.show()

    # 3. Churn por adulto mayor
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Senior Citizen', hue='Etiqueta Churn', data=df)
    plt.title('Churn (Baja de clientes) por Adulto mayor')
    plt.xlabel('Adulto mayor')
    plt.ylabel('Cantidad')
    plt.savefig('churn_by_seniorcitizen.png')
    plt.show()

    # 4. Churn por tipo de contrato
    plt.figure(figsize=(10, 8))
    sns.countplot(x='Contract', hue='Etiqueta Churn', data=df, order=['Mensual', 'Un año', 'Dos años'])
    plt.title('Churn (Baja de clientes) por Tipo de Contrato')
    plt.xlabel('Tipo de Contrato')
    plt.ylabel('Cantidad')
    plt.savefig('churn_by_contract.png')
    plt.show()

    # 5. Churn por servicio de internet
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Internet Service', hue='Etiqueta Churn', data=df)
    plt.title('Churn (Baja de clientes) por Tipo de Servicio de Internet')
    plt.xlabel('Servicio de Internet')
    plt.ylabel('Cantidad')
    plt.savefig('churn_by_internet_service.png')
    plt.show()

    # 6. Churn por servicio de telefonia
    plt.figure(figsize=(12, 8))
    sns.countplot(x='Phone Service', hue='Etiqueta Churn', data=df)
    plt.title('Churn (Baja de clientes) por Tipo de Servicio de Telefonía')
    plt.xlabel('Servicio de Telefonia')
    plt.ylabel('Cantidad')
    plt.savefig('churn_by_telefonia_service.png')
    plt.show()

    # 7. Distribución de meses de permanencia
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Tenure Months', hue='Etiqueta Churn', bins=30, kde=True, multiple='stack')
    plt.title('Distribución de Meses de Permanencia por Estado de Churn')
    plt.xlabel('Meses de Permanencia')
    plt.ylabel('Cantidad de Clientes')
    plt.savefig('tenure_distribution.png')
    plt.show()

    # 8. Cargos mensuales vs Churn (Boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Etiqueta Churn', y='Monthly Charges', data=df)
    plt.title('Distribución de Cargos Mensuales por Estado de Churn')
    plt.xlabel('Estado de Churn')
    plt.ylabel('Cargos Mensuales ($)')
    plt.savefig('monthly_charges_vs_churn.png')
    plt.show()

    # 9. Mapa de calor de correlación (variables numéricas)
    numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']
    df_renombrado = df[numeric_cols].rename(columns={
        'Tenure Months': 'Meses de antigüedad',
        'Monthly Charges': 'Cargos mensuales',
        'Total Charges': 'Cargos totales',
        'Churn Score': 'Índice de abandono',
        'CLTV': 'Valor vida cliente'
    })
    plt.figure(figsize=(16, 8))
    sns.heatmap(df_renombrado.corr(), annot=True, cmap='coolwarm', center=0,
                cbar_kws={'location': 'right'}, fmt=".2f", linewidths=.5)
    plt.title('Mapa de Calor de Correlación entre Variables Numéricas')
    plt.savefig('correlation_heatmap.png')
    plt.show()

    # 10. Razones principales de churn (top 10)
    plt.figure(figsize=(16, 8))
    reason_counts = df[df['Etiqueta Churn'] == 'Si']['Churn Reason'].value_counts().head(10)
    reason_counts.plot(kind='barh', color='teal')
    plt.title('Top 10 Razones de Churn')
    plt.xlabel('Cantidad de Clientes')
    plt.ylabel('Razón de Churn')
    plt.savefig('top_churn_reasons.png')
    plt.show()

    # 11. Churn por método de pago
    plt.figure(figsize=(14, 7))
    sns.countplot(y='Payment Method', hue='Etiqueta Churn', data=df)
    plt.title('Churn por Método de Pago')
    plt.xlabel('Cantidad')
    plt.ylabel('Método de Pago')
    plt.savefig('churn_by_payment_method.png')
    plt.show()

    # 12. Distribución de CLTV por Churn (boxplot)
    plt.figure(figsize=(9, 8))
    sns.boxplot(x='Etiqueta Churn', y='CLTV', data=df)
    plt.title('Distribución de CLTV por Estado de Churn')
    plt.xlabel('Estado de Churn')
    plt.ylabel('CLTV (Valor de vida del cliente)')
    plt.savefig('cltv_vs_churn.png')
    plt.show()

# --- Funciones para generación de reportes (desde Avance proyecto) ---

def generate_demographic_report(df):
    report = {
        "total_customers": len(df),
        "churn_rate": df['Etiqueta Churn'].value_counts(normalize=True).get('Si', 0) * 100,
        "gender_distribution": df['Gender'].value_counts(normalize=True).mul(100),
        "senior_citizen_percentage": df['Senior Citizen'].value_counts(normalize=True).get('Si', 0) * 100,
        "partner_percentage": df['Partner'].value_counts(normalize=True).get('Si', 0) * 100,
        "dependents_percentage": df['Dependents'].value_counts(normalize=True).get('Si', 0) * 100,
        "average_tenure": df['Tenure Months'].mean(),
        "average_monthly_charges": df['Monthly Charges'].mean(),
        "average_cltv": df['CLTV'].mean()
    }
    return report

def generate_churn_report(df):
    churned = df[df['Etiqueta Churn'] == 'Si']
    not_churned = df[df['Etiqueta Churn'] == 'No']
    report = {
        "total_churned": len(churned),
        "churn_rate": (len(churned) / len(df) * 100) if len(df) > 0 else 0,
        "top_reasons": churned['Churn Reason'].value_counts().head(5).to_dict(),
        "avg_tenure_churned": churned['Tenure Months'].mean(),
        "avg_tenure_not_churned": not_churned['Tenure Months'].mean(),
        "avg_monthly_charges_churned": churned['Monthly Charges'].mean(),
        "avg_monthly_charges_not_churned": not_churned['Monthly Charges'].mean(),
        "avg_cltv_churned": churned['CLTV'].mean(),
        "avg_cltv_not_churned": not_churned['CLTV'].mean()
    }
    return report

def imprimir_reportes(demographic_report, churn_report):
    print("\n=== Reporte Demográfico ===")
    for key, value in demographic_report.items():
        if hasattr(value, 'items'):
            # Si es Serie, mostrar valores
            print(f"{key.replace('_', ' ').title()}:")
            if isinstance(value, pd.Series):
                for k, v in value.items():
                    print(f"  - {k}: {v:.2f}%")
            else:
                print(value)
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

    print("\n=== Reporte de Churn ===")
    for key, value in churn_report.items():
        if isinstance(value, dict):
            print(f"\n{key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f" - {k}: {v}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

# --- Análisis estadístico detallado y modelado desde machine.py ---

def analisis_descriptivo(df):
    print("\n=== ANÁLISIS DESCRIPTIVO ===\n")
    numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']

    stats_df = pd.DataFrame(index=numeric_cols, columns=[
        'Media', 'Mediana', 'Moda', 'Varianza', 'Desv. Estándar',
        'Coef. Variación', 'Asimetría', 'Curtosis', 'Mínimo', 'Máximo', 'Rango'
    ])
    for col in numeric_cols:
        stats_df.loc[col] = [
            df[col].mean(),
            df[col].median(),
            df[col].mode()[0] if not df[col].mode().empty else np.nan,
            df[col].var(),
            df[col].std(),
            df[col].std() / df[col].mean() if df[col].mean() != 0 else np.nan,
            df[col].skew(),
            df[col].kurtosis(),
            df[col].min(),
            df[col].max(),
            df[col].max() - df[col].min()
        ]

    # Visualizaciones
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df[numeric_cols], orient='h')
    plt.title('Distribución de Variables Numéricas')
    plt.savefig('1_distribucion_numericas.png')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.delaxes(axes[1, 2])
    for i, col in enumerate(numeric_cols):
        ax = axes[i//3, i%3]
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribución de {col}')
    plt.tight_layout()
    plt.savefig('2_histogramas_numericas.png')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.delaxes(axes[1, 2])
    for i, col in enumerate(numeric_cols):
        ax = axes[i//3, i%3]
        sns.boxplot(x='Etiqueta Churn', y=col, data=df, ax=ax)
        ax.set_title(f'{col} por Estado de Churn')
    plt.tight_layout()
    plt.savefig('3_boxplot_churn.png')
    plt.show()

    return stats_df.round(2)

def analisis_inferencial(df):
    print("\n=== ANÁLISIS INFERENCIAL ===\n")
    results = {}
    numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']

    # Intervalos de confianza al 95%
    conf_intervals = {}
    for col in numeric_cols:
        n = len(df[col])
        mean = df[col].mean()
        std = df[col].std()
        conf_int = stats.norm.interval(0.95, loc=mean, scale=std/np.sqrt(n))
        conf_intervals[col] = conf_int

    plt.figure(figsize=(12, 6))
    means = [df[col].mean() for col in numeric_cols]
    lowers = [conf_intervals[col][0] for col in numeric_cols]
    uppers = [conf_intervals[col][1] for col in numeric_cols]
    plt.errorbar(x=numeric_cols, y=means,
                 yerr=[means[i]-lowers[i] for i in range(len(means))],
                 fmt='o', color='darkblue', ecolor='lightblue',
                 elinewidth=3, capsize=0)
    plt.title('Medias con Intervalos de Confianza del 95%')
    plt.ylabel('Valor Promedio')
    plt.savefig('4_intervalos_confianza.png')
    plt.show()

    results['intervalos_confianza'] = conf_intervals

    # Pruebas t para comparar Churn vs No Churn
    t_test_results = {}
    for col in numeric_cols:
        grupo_churn = df[df['Churn Numeric'] == 1][col]
        grupo_no_churn = df[df['Churn Numeric'] == 0][col]
        t_stat, p_val = stats.ttest_ind(grupo_churn, grupo_no_churn, equal_var=False)
        t_test_results[col] = {'t-statistic': t_stat, 'p-value': p_val}

    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_cols):
        plt.subplot(1, 3, i+1)
        sns.barplot(x='Etiqueta Churn', y=col, data=df, ci='sd')
        plt.title(f'{col}\n(p={t_test_results[col]["p-value"]:.4f})')
    plt.tight_layout()
    plt.savefig('5_comparacion_medias.png')
    plt.show()

    results['pruebas_t'] = t_test_results

    # ANOVA de Monthly Charges por Contract
    try:
        df_anova = df[['Monthly Charges', 'Contract']].dropna()
        model = ols('Q("Monthly Charges") ~ C(Contract)', data=df_anova).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        results['anova'] = anova_table

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Contract', y='Monthly Charges', data=df, order=['Mensual', 'Un año', 'Dos años'])
        plt.title('Distribución de Cargos Mensuales por Tipo de Contrato')
        plt.savefig('6_anova_contrato.png')
        plt.show()
    except Exception as e:
        print(f"Error en ANOVA: {str(e)}")
        results['anova'] = None

    # Correlación de Pearson
    numeric_all = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV', 'Churn Numeric']
    corr_matrix = df[numeric_all].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Matriz de Correlación de Pearson')
    plt.savefig('7_correlacion_pearson.png')
    plt.show()

    results['correlacion_pearson'] = corr_matrix

    return results

def modelado_predictivo(df):
    print("\n=== MODELADO PREDICTIVO ===\n")
    results = {}

    categorical_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                        'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                        'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']

    le = LabelEncoder()
    for col in categorical_cols:
        df[col+'_encoded'] = le.fit_transform(df[col])

    features_encoded = [col for col in df.columns if col.endswith('_encoded')]
    features_num = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']
    features = features_encoded + features_num

    X = df[features]
    y = df['Churn Numeric']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Modelo Regresión Logística
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    # Modelo Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    results['logistic_regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'classification_report': classification_report(y_test, y_pred_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_prob_lr)
    }

    results['random_forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'classification_report': classification_report(y_test, y_pred_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_prob_rf),
        'feature_importances': pd.Series(rf.feature_importances_, index=X.columns)
    }

    # Gráfico importancia características Random Forest
    plt.figure(figsize=(12, 8))
    results['random_forest']['feature_importances'].nlargest(10).plot(kind='barh')
    plt.title('Top 10 Características Más Importantes (Random Forest)')
    plt.savefig('8_importancia_caracteristicas.png')
    plt.show()

    # Matrices de confusión
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(results['logistic_regression']['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Matriz de Confusión - Regresión Logística')
    ax1.set_xlabel('Predicho')
    ax1.set_ylabel('Real')
    sns.heatmap(results['random_forest']['confusion_matrix'], annot=True, fmt='d', cmap='Greens', ax=ax2)
    ax2.set_title('Matriz de Confusión - Random Forest')
    ax2.set_xlabel('Predicho')
    ax2.set_ylabel('Real')
    plt.tight_layout()
    plt.savefig('9_matrices_confusion.png')
    plt.show()

    # Curva ROC
    plt.figure(figsize=(10, 8))
    for model, y_prob, color, name in zip([lr, rf], [y_prob_lr, y_prob_rf], ['blue', 'green'],
                                          ['Regresión Logística', 'Random Forest']):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC - Comparación de Modelos')
    plt.legend()
    plt.savefig('10_curva_roc.png')
    plt.show()

    return results

# --- Función Principal Integrada ---

def main():
    # Cargar datos
    df = cargar_datos()
    if df is None:
        print("No se pudo cargar el dataset. Terminando ejecución.")
        return

    # Análisis Exploratorio básico
    analisis_exploratorio_basico(df)

    # Generar reportes con funciones del avance proyecto
    demographic_report = generate_demographic_report(df)
    churn_report = generate_churn_report(df)
    imprimir_reportes(demographic_report, churn_report)

    # Análisis descriptivo detallado con visualizaciones
    stats_results = analisis_descriptivo(df)
    print("\nEstadísticas Descriptivas:")
    print(stats_results)

    # Análisis inferencial estadístico
    inferencial_results = analisis_inferencial(df)
    print("\nIntervalos de Confianza (95%):")
    for col, interval in inferencial_results['intervalos_confianza'].items():
        print(f"{col}: [{interval[0]:.2f}, {interval[1]:.2f}]")
    print("\nPruebas t (Churn vs No Churn):")
    for col, result in inferencial_results['pruebas_t'].items():
        print(f"{col}: t = {result['t-statistic']:.2f}, p = {result['p-value']:.4f}")
    print("\nANOVA (Monthly Charges por Tipo de Contrato):")
    if inferencial_results['anova'] is not None:
        print(inferencial_results['anova'])
    else:
        print("ANOVA no realizado.")

    # Modelado predictivo y evaluación
    ml_results = modelado_predictivo(df)
    print("\nResultados Regresión Logística:")
    print(f"Accuracy: {ml_results['logistic_regression']['accuracy']:.2f}")
    print(ml_results['logistic_regression']['classification_report'])

    print("\nResultados Random Forest:")
    print(f"Accuracy: {ml_results['random_forest']['accuracy']:.2f}")
    print(ml_results['random_forest']['classification_report'])

    # Guardar reportes de texto
    with open('demographic_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Reporte Demográfico ===\n")
        for key, value in demographic_report.items():
            if hasattr(value, 'items'):
                f.write(f"{key.replace('_', ' ').title()}:\n")
                if isinstance(value, pd.Series):
                    for k, v in value.items():
                        f.write(f"  - {k}: {v:.2f}%\n")
                else:
                    f.write(str(value) + "\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}\n")

    with open('churn_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Reporte de Churn ===\n")
        for key, value in churn_report.items():
            if isinstance(value, dict):
                f.write(f"\n{key.replace('_', ' ').title()}:\n")
                for k, v in value.items():
                    f.write(f" - {k}: {v}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}\n")

    with open('resultados_analisis.txt', 'w', encoding='utf-8') as f:
        f.write("=== RESULTADOS DEL ANÁLISIS ===\n\n")
        f.write("1. ESTADÍSTICAS DESCRIPTIVAS:\n")
        f.write(stats_results.to_string())
        f.write("\n\n2. ANÁLISIS INFERENCIAL:\n")
        f.write("Intervalos de Confianza (95%):\n")
        for col, interval in inferencial_results['intervalos_confianza'].items():
            f.write(f"{col}: [{interval[0]:.2f}, {interval[1]:.2f}]\n")
        f.write("\nPruebas t (Churn vs No Churn):\n")
        for col, result in inferencial_results['pruebas_t'].items():
            f.write(f"{col}: t = {result['t-statistic']:.2f}, p = {result['p-value']:.4f}\n")
        f.write("\nANOVA:\n")
        f.write(str(inferencial_results['anova']) if inferencial_results['anova'] is not None else "ANOVA no realizado.")
        f.write("\n\n3. MODELADO PREDICTIVO:\n")
        f.write("Regresión Logística:\n")
        f.write(f"Accuracy: {ml_results['logistic_regression']['accuracy']:.2f}\n")
        f.write(ml_results['logistic_regression']['classification_report'])
        f.write("\nRandom Forest:\n")
        f.write(f"Accuracy: {ml_results['random_forest']['accuracy']:.2f}\n")
        f.write(ml_results['random_forest']['classification_report'])

    print("\nAnálisis completado. Gráficos y reportes guardados en el directorio actual.")

if __name__ == "__main__":
    main()
