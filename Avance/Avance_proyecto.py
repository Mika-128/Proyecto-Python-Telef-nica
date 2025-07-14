"""Avance de Proyecto de Modelo Predictivo de Abandono de Clientes (Churn) en Servicios de Telecomuniaciones
 (Telefonía Fija e Internet)
Dataset: https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
Archivo: Telco_customer_churn.xlsx
Librerias: Pandas, Numpy, matplotlib y seaborn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 7)
sns.set_palette("husl")

# Cargar los datos
df = pd.read_excel('Telco_customer_churn.xlsx', sheet_name='Telco_customer_churn')

# Limpieza inicial de datos
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'] = df['Total Charges'].fillna(0)

# Análisis exploratorio básico
print("\nResumen estadístico de las columnas numéricas:")
print(df.describe())

print("\nDistribución de la variable objetivo (Etiqueta Churn):")
print(df['Etiqueta Churn'].value_counts())

# Visualizaciones

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

# 6. Churn por servicio de telefónia
plt.figure(figsize=(12, 8))
sns.countplot(x='Phone Service', hue='Etiqueta Churn', data=df)
plt.title('Churn (Baja de clientes) por Tipo de Servicio de Telefonía')
plt.xlabel('Servicio de Telefonia')
plt.ylabel('Cantidad')
plt.savefig('churn_by_telefonia_service.png')
plt.show()

# 5. Churn por servicio de internet
plt.figure(figsize=(12, 8))
sns.countplot(x='Internet Service', hue='Etiqueta Churn', data=df)
plt.title('Churn (Baja de clientes) por Tipo de Servicio de Internet')
plt.xlabel('Servicio de Internet')
plt.ylabel('Cantidad')
plt.savefig('churn_by_internet_service.png')
plt.show()

# 6. Distribución de meses de permanencia
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='Tenure Months', hue='Etiqueta Churn', bins=30, kde=True, multiple='stack')
plt.title('Distribución de Meses de Permanencia por Estado de Churn (Baja de clientes)')
plt.xlabel('Meses de Permanencia')
plt.ylabel('Cantidad de Clientes')
plt.savefig('tenure_distribution.png')
plt.show()

# 7. Cargos mensuales vs Churn
plt.figure(figsize=(12, 8))
sns.boxplot(x='Etiqueta Churn', y='Monthly Charges', data=df)
plt.title('Distribución de Cargos Mensuales por Estado de Churn (Baja de clientes)')
plt.xlabel('Estado de Churn')
plt.ylabel('Cargos Mensuales ($)')
plt.savefig('monthly_charges_vs_churn.png')
plt.show()

# 8. Mapa de calor de correlación
numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV']

# Diccionario para traducir/renombrar las columnas
nombres_espanol = {
    'Tenure Months': 'Meses de antigüedad',
    'Monthly Charges': 'Cargos mensuales',
    'Total Charges': 'Cargos totales',
    'Churn Score': 'Índice de abandono',
    'CLTV': 'CLTV Valor vida cliente'  
}
# Crear una copia del DataFrame con las columnas renombradas
df_renombrado = df[numeric_cols].rename(columns=nombres_espanol)

plt.figure(figsize=(16, 8))
plt.subplots_adjust(left=0.3)
sns.heatmap(df_renombrado.corr(), annot=True, cmap='coolwarm', center=0, 
            cbar_kws={'location': 'right'}, fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor de Correlación entre Variables Numéricas')
plt.savefig('correlation_heatmap.png')
plt.show()

# 9. Razones principales de churn
plt.figure(figsize=(16, 8))
reason_counts = df[df['Etiqueta Churn'] == 'Si']['Churn Reason'].value_counts().head(10)
plt.subplots_adjust(left=0.3)
reason_counts.plot(kind='barh', color='teal')
plt.title('Top 10 Razones de Churn (Baja de clientes)')
plt.xlabel('Cantidad de Clientes')
plt.ylabel('Razón de Churn (Baja de clientes)')
plt.savefig('top_churn_reasons.png')
plt.show()

# 10. Churn por método de pago
plt.figure(figsize=(14, 7))
plt.subplots_adjust(left=0.3)
sns.countplot(y='Payment Method', hue='Etiqueta Churn', data=df)
plt.title('Churn (Baja de clientes) por Método de Pago')
plt.xlabel('Cantidad')
plt.ylabel('Método de Pago')
plt.savefig('churn_by_payment_method.png')
plt.show()

# 11. Distribución de CLTV por Churn
plt.figure(figsize=(9, 8))
plt.subplots_adjust(left=0.3)
sns.boxplot(x='Etiqueta Churn', y='CLTV', data=df)
plt.title('Distribución de CLTV por Estado de Churn (Baja de clientes)')
plt.xlabel('Estado de Churn')
plt.ylabel('CLTV (Valor de vida del cliente)')
plt.savefig('cltv_vs_churn.png')
plt.show()

# Generación de reportes

# 1Función para generar reporte demográfico
def generate_demographic_report(df):
    report = {
        "total_customers": len(df),
        "churn_rate": df['Etiqueta Churn'].value_counts(normalize=True)['Si'] * 100,
        "gender_distribution": df['Gender'].value_counts(normalize=True) * 100,
        "senior_citizen_percentage": df['Senior Citizen'].value_counts(normalize=True)['Si'] * 100,
        "partner_percentage": df['Partner'].value_counts(normalize=True)['Si'] * 100,
        "dependents_percentage": df['Dependents'].value_counts(normalize=True)['Si'] * 100,
        "average_tenure": df['Tenure Months'].mean(),
        "average_monthly_charges": df['Monthly Charges'].mean(),
        "average_cltv": df['CLTV'].mean()
    }
    return report

# 2Función para generar reporte de churn
def generate_churn_report(df):
    churned = df[df['Etiqueta Churn'] == 'Si']
    not_churned = df[df['Etiqueta Churn'] == 'No']
    
    report = {
        "total_churned": len(churned),
        "churn_rate": len(churned) / len(df) * 100,
        "top_reasons": churned['Churn Reason'].value_counts().head(5).to_dict(),
        "avg_tenure_churned": churned['Tenure Months'].mean(),
        "avg_tenure_not_churned": not_churned['Tenure Months'].mean(),
        "avg_monthly_charges_churned": churned['Monthly Charges'].mean(),
        "avg_monthly_charges_not_churned": not_churned['Monthly Charges'].mean(),
        "avg_cltv_churned": churned['CLTV'].mean(),
        "avg_cltv_not_churned": not_churned['CLTV'].mean()
    }
    return report

# 3Generar y mostrar reportes
demographic_report = generate_demographic_report(df)
churn_report = generate_churn_report(df)

print("\n=== Reporte Demográfico ===")
for key, value in demographic_report.items():
    print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

print("\n=== Reporte de Churn ===")
for key, value in churn_report.items():
    if isinstance(value, dict):
        print(f"\n{key.replace('_', ' ').title()}:")
        for reason, count in value.items():
            print(f"  - {reason}: {count}")
    else:
        print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}")

# 4Guardar reportes en archivos
with open('demographic_report.txt', 'w') as f:
    f.write("=== Reporte Demográfico ===\n")
    for key, value in demographic_report.items():
        f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}\n")

with open('churn_report.txt', 'w') as f:
    f.write("=== Reporte de Churn ===\n")
    for key, value in churn_report.items():
        if isinstance(value, dict):
            f.write(f"\n{key.replace('_', ' ').title()}:\n")
            for reason, count in value.items():
                f.write(f"  - {reason}: {count}\n")
        else:
            f.write(f"{key.replace('_', ' ').title()}: {value:.2f}\n" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value}\n")

print("\nAnálisis completado. Gráficos y reportes guardados en el directorio actual.")