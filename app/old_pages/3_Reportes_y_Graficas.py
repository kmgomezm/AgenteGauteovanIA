# pages/3_Reportes_y_Graficas.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.plots import plot_counts
from datetime import datetime
import numpy as np

st.set_page_config(
    page_title="Reportes y Gráficas hh", 
    page_icon="📈",
    layout="wide"
)

# ===== REPORTES Y GRÁFICAS =====
st.title("📈 Reportes y Gráficas del Corpus")
st.markdown("Análisis estadístico y visualizaciones del corpus de columnas de opinión")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    st.markdown("**Opciones de visualización:**")
    chart_style = st.selectbox(
        "Estilo de gráficas:",
        ["default", "seaborn", "ggplot", "dark_background"],
        help="Estilo visual para las gráficas"
    )
    
    color_palette = st.selectbox(
        "Paleta de colores:",
        ["tab10", "viridis", "plasma", "Set3", "pastel"],
        help="Paleta de colores para las visualizaciones"
    )
    
    st.markdown("---")
    st.page_link("streamlit_app.py", label="💬 Volver al Chat")
    st.page_link("pages/1_Buscador_Semantico.py", label="🔍 Búsqueda Semántica")
    st.page_link("pages/2_Analisis_NLP.py", label="📊 Análisis NLP")

# Configurar estilo de matplotlib
plt.style.use(chart_style)

# Cargar datos
@st.cache_data
def load_corpus_data():
    """Carga los datos del corpus procesado"""
    try:
        df = pd.read_parquet("data/processed/chunks.parquet")
        return df, None
    except FileNotFoundError:
        return None, "Archivo chunks.parquet no encontrado en data/processed/"
    except Exception as e:
        return None, f"Error al cargar datos: {str(e)}"

# Función para generar estadísticas
def generate_statistics(df):
    """Genera estadísticas descriptivas del corpus"""
    stats = {}
    
    # Estadísticas básicas
    stats['total_chunks'] = len(df)
    stats['total_documents'] = df['doc_id'].nunique()
    stats['total_authors'] = df['autor'].nunique() 
    stats['total_newspapers'] = df['diario'].nunique()
    
    # Estadísticas de texto
    df['chunk_length'] = df['chunk'].astype(str).apply(len)
    df['word_count'] = df['chunk'].astype(str).apply(lambda x: len(x.split()))
    
    stats['avg_chunk_length'] = df['chunk_length'].mean()
    stats['avg_word_count'] = df['word_count'].mean()
    stats['total_words'] = df['word_count'].sum()
    
    # Fechas
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    stats['date_range'] = {
        'start': df['fecha'].min(),
        'end': df['fecha'].max()
    }
    
    # Top autores y medios
    stats['top_authors'] = df['autor'].value_counts().head(10)
    stats['top_newspapers'] = df['diario'].value_counts()
    
    return stats

# Cargar y procesar datos
df, error = load_corpus_data()

if error:
    st.error(f"❌ {error}")
    
    # Opción para cargar archivo manualmente
    st.markdown("### 📁 Cargar archivo manualmente")
    uploaded_file = st.file_uploader(
        "Sube el archivo chunks.parquet:",
        type=['parquet'],
        help="Carga el archivo procesado del corpus"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_parquet(uploaded_file)
            st.success("✅ Archivo cargado exitosamente")
        except Exception as e:
            st.error(f"Error al procesar archivo: {e}")
            st.stop()
    else:
        st.stop()

# Generar estadísticas
stats = generate_statistics(df)

# ===== DASHBOARD PRINCIPAL =====
st.markdown("## 📊 Resumen Ejecutivo")

# Métricas principales
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📄 Total Fragmentos", 
        f"{stats['total_chunks']:,}",
        help="Número total de fragmentos de texto"
    )

with col2:
    st.metric(
        "📰 Documentos", 
        f"{stats['total_documents']:,}",
        help="Número total de columnas de opinión"
    )

with col3:
    st.metric(
        "✍️ Autores", 
        f"{stats['total_authors']:,}",
        help="Número de autores únicos"
    )

with col4:
    st.metric(
        "🏢 Medios", 
        f"{stats['total_newspapers']:,}",
        help="Número de medios de comunicación"
    )

# Segunda fila de métricas
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric(
        "💬 Total Palabras", 
        f"{stats['total_words']:,}",
        help="Número total de palabras en el corpus"
    )

with col6:
    st.metric(
        "📏 Promedio Palabras/Fragmento", 
        f"{stats['avg_word_count']:.1f}",
        help="Promedio de palabras por fragmento"
    )

with col7:
    st.metric(
        "📅 Período", 
        f"{stats['date_range']['start'].year}-{stats['date_range']['end'].year}",
        help="Rango temporal del corpus"
    )

with col8:
    años_span = stats['date_range']['end'].year - stats['date_range']['start'].year + 1
    st.metric(
        "⏱️ Duración", 
        f"{años_span} años",
        help="Duración temporal del corpus"
    )

# ===== VISUALIZACIONES =====
st.markdown("---")
st.markdown("## 📈 Visualizaciones")

# Tabs para diferentes tipos de análisis
tab1, tab2, tab3, tab4 = st.tabs([
    "📰 Por Medio", 
    "✍️ Por Autor", 
    "📅 Temporal", 
    "📊 Distribuciones"
])

with tab1:
    st.markdown("### 📰 Análisis por Medio de Comunicación")
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Gráfico de barras
        newspaper_counts = stats['top_newspapers']
        bars = ax.bar(range(len(newspaper_counts)), newspaper_counts.values, 
                     color=plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(newspaper_counts))))
        
        ax.set_xlabel('Medios de Comunicación')
        ax.set_ylabel('Número de Fragmentos')
        ax.set_title('Distribución por Medio de Comunicación')
        ax.set_xticks(range(len(newspaper_counts)))
        ax.set_xticklabels(newspaper_counts.index, rotation=45, ha='right')
        
        # Agregar valores en las barras
        for bar, value in zip(bars, newspaper_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(newspaper_counts.values),
                   f'{value:,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_table:
        st.markdown("**📋 Tabla de medios:**")
        newspaper_df = pd.DataFrame({
            'Medio': newspaper_counts.index,
            'Fragmentos': newspaper_counts.values,
            'Porcentaje': (newspaper_counts.values / newspaper_counts.sum() * 100).round(1)
        })
        st.dataframe(newspaper_df, use_container_width=True)
        
        # Descargar tabla
        csv = newspaper_df.to_csv(index=False)
        st.download_button(
            label="📁 Descargar CSV",
            data=csv,
            file_name="distribucion_medios.csv",
            mime="text/csv"
        )

with tab2:
    st.markdown("### ✍️ Análisis por Autor")
    
    # Control para número de autores a mostrar
    n_authors = st.slider("Número de autores a mostrar:", 5, 20, 10)
    
    top_authors = stats['top_authors'].head(n_authors)
    
    col_chart, col_table = st.columns([2, 1])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Gráfico horizontal para mejor legibilidad
        bars = ax.barh(range(len(top_authors)), top_authors.values,
                      color=plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(top_authors))))
        
        ax.set_ylabel('Autores')
        ax.set_xlabel('Número de Fragmentos')
        ax.set_title(f'Top {n_authors} Autores Más Prolíficos')
        ax.set_yticks(range(len(top_authors)))
        ax.set_yticklabels(top_authors.index)
        
        # Valores en las barras
        for i, (bar, value) in enumerate(zip(bars, top_authors.values)):
            ax.text(value + 0.01*max(top_authors.values), i,
                   f'{value:,}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_table:
        st.markdown("**📋 Tabla de autores:**")
        authors_df = pd.DataFrame({
            'Autor': top_authors.index,
            'Fragmentos': top_authors.values,
            'Porcentaje': (top_authors.values / df['autor'].value_counts().sum() * 100).round(1)
        })
        st.dataframe(authors_df, use_container_width=True, height=400)

with tab3:
    st.markdown("### 📅 Análisis Temporal")
    
    # Preparar datos temporales
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
    df['año_mes'] = df['fecha'].dt.to_period('M')
    
    # Filtrar datos válidos
    df_temporal = df.dropna(subset=['fecha'])
    
    if len(df_temporal) > 0:
        col_year, col_month = st.columns(2)
        
        with col_year:
            st.markdown("#### 📅 Distribución por Año")
            yearly_counts = df_temporal['año'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
            ax.fill_between(yearly_counts.index, yearly_counts.values, alpha=0.3)
            ax.set_xlabel('Año')
            ax.set_ylabel('Número de Fragmentos')
            ax.set_title('Evolución Temporal por Año')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col_month:
            st.markdown("#### 📅 Distribución por Mes")
            monthly_counts = df_temporal['mes'].value_counts().sort_index()
            
            meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                    'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(1, 13), [monthly_counts.get(i, 0) for i in range(1, 13)])
            ax.set_xlabel('Mes')
            ax.set_ylabel('Número de Fragmentos')
            ax.set_title('Distribución por Mes del Año')
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(meses)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Heatmap temporal (si hay suficientes datos)
        if len(df_temporal) > 100:
            st.markdown("#### 🔥 Mapa de Calor Temporal")
            
            # Crear matriz año-mes
            pivot_data = df_temporal.groupby(['año', 'mes']).size().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
            ax.set_xlabel('Mes')
            ax.set_ylabel('Año')
            ax.set_title('Distribución Temporal - Mapa de Calor')
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("⚠️ No hay datos temporales válidos para mostrar")

with tab4:
    st.markdown("### 📊 Distribuciones Estadísticas")
    
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        st.markdown("#### 📏 Distribución de Longitud de Fragmentos")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['chunk_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(df['chunk_length'].mean(), color='red', linestyle='--', 
                  label=f'Media: {df["chunk_length"].mean():.0f}')
        ax.axvline(df['chunk_length'].median(), color='green', linestyle='--', 
                  label=f'Mediana: {df["chunk_length"].median():.0f}')
        ax.set_xlabel('Longitud del Fragmento (caracteres)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Longitud de Fragmentos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_hist2:
        st.markdown("#### 💬 Distribución de Número de Palabras")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['word_count'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(df['word_count'].mean(), color='blue', linestyle='--', 
                  label=f'Media: {df["word_count"].mean():.1f}')
        ax.axvline(df['word_count'].median(), color='orange', linestyle='--', 
                  label=f'Mediana: {df["word_count"].median():.1f}')
        ax.set_xlabel('Número de Palabras')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Palabras por Fragmento')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Estadísticas descriptivas
    st.markdown("#### 📈 Estadísticas Descriptivas")
    
    descriptive_stats = pd.DataFrame({
        'Métrica': ['Longitud (caracteres)', 'Número de palabras'],
        'Media': [df['chunk_length'].mean(), df['word_count'].mean()],
        'Mediana': [df['chunk_length'].median(), df['word_count'].median()],
        'Desv. Estándar': [df['chunk_length'].std(), df['word_count'].std()],
        'Mínimo': [df['chunk_length'].min(), df['word_count'].min()],
        'Máximo': [df['chunk_length'].max(), df['word_count'].max()],
        'Q25': [df['chunk_length'].quantile(0.25), df['word_count'].quantile(0.25)],
        'Q75': [df['chunk_length'].quantile(0.75), df['word_count'].quantile(0.75)]
    })
    
    # Redondear números
    for col in ['Media', 'Mediana', 'Desv. Estándar', 'Q25', 'Q75']:
        descriptive_stats[col] = descriptive_stats[col].round(1)
    
    st.dataframe(descriptive_stats, use_container_width=True)

# ===== EXPORTAR REPORTES =====
st.markdown("---")
st.markdown("## 💾 Exportar Reportes")

col_export1, col_export2, col_export3 = st.columns(3)

with col_export1:
    # Resumen estadístico
    summary_report = f"""
# Reporte Estadístico del Corpus Gauteovan

## Resumen Ejecutivo
- **Total de fragmentos:** {stats['total_chunks']:,}
- **Total de documentos:** {stats['total_documents']:,}
- **Autores únicos:** {stats['total_authors']:,}
- **Medios de comunicación:** {stats['total_newspapers']:,}
- **Total de palabras:** {stats['total_words']:,}
- **Período:** {stats['date_range']['start'].strftime('%Y-%m-%d')} a {stats['date_range']['end'].strftime('%Y-%m-%d')}

## Estadísticas de Texto
- **Promedio palabras/fragmento:** {stats['avg_word_count']:.1f}
- **Promedio caracteres/fragmento:** {stats['avg_chunk_length']:.1f}

## Top 10 Autores
{chr(10).join([f"- {autor}: {count:,} fragmentos" for autor, count in stats['top_authors'].items()])}

## Medios de Comunicación
{chr(10).join([f"- {medio}: {count:,} fragmentos" for medio, count in stats['top_newspapers'].items()])}

---
Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    st.download_button(
        label="📄 Descargar Reporte Estadístico",
        data=summary_report,
        file_name=f"reporte_corpus_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )

with col_export2:
    # Exportar datos agregados
    export_data = pd.DataFrame({
        'Medio': stats['top_newspapers'].index,
        'Fragmentos': stats['top_newspapers'].values,
        'Porcentaje': (stats['top_newspapers'].values / stats['top_newspapers'].sum() * 100).round(2)
    })
    
    csv_data = export_data.to_csv(index=False)
    st.download_button(
        label="📊 Datos Agregados (CSV)",
        data=csv_data,
        file_name=f"datos_agregados_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

with col_export3:
    # Estadísticas descriptivas
    descriptive_export = descriptive_stats.to_csv(index=False)
    st.download_button(
        label="📈 Estadísticas Descriptivas",
        data=descriptive_export,
        file_name=f"estadisticas_descriptivas_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("### ℹ️ Información")
st.markdown("""
Este dashboard proporciona un análisis completo del corpus de columnas de opinión:

- **📊 Métricas generales**: Conteos básicos y estadísticas del corpus
- **📈 Análisis temporal**: Distribución de contenido a lo largo del tiempo
- **👥 Análisis de autoría**: Productividad y distribución por autores
- **📰 Análisis por medios**: Representación de diferentes periódicos
- **📊 Distribuciones**: Características estadísticas del texto

**Datos actualizados:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")