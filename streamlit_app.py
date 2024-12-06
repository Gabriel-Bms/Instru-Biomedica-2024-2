import requests
import plotly.graph_objects as go
import streamlit as st
import time
import numpy as np
import pickle
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks
import pandas as pd
import hrvanalysis
from hrvanalysis import get_csi_cvi_features, get_frequency_domain_features
from hrvanalysis import get_geometrical_features, get_poincare_plot_features
from hrvanalysis import get_sampen, get_time_domain_features


st.set_page_config(page_title="main", layout="wide",initial_sidebar_state="collapsed",)
firebase_url = "https://esp32-ib-default-rtdb.firebaseio.com/testeo.json"
hide_streamlit_style = """
<style>
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with open("model_xgb_new.pkl", 'rb') as file:
    model = pickle.load(file)

def iniciar_grabacion():
    st.session_state["grabando"] = True
    st.session_state["datos_grabados"] = []
    st.sidebar.success("Iniciando Grabación")
def finalizar_grabacion():
    st.session_state["grabando"] = False
    if st.session_state["datos_grabados"]:
        with open("grabacion_datos.txt", "w") as file:
            for linea in st.session_state["datos_grabados"]:
                file.write(linea + "\n")
        st.sidebar.success("Grabación finalizada. Archivo generado.")
        with open("grabacion_datos.txt", "rb") as file:
            st.sidebar.download_button(
                label="Descargar Archivo",
                data=file,
                file_name="grabacion_datos.txt",
                mime="text/plain",
            )
    else:
        st.sidebar.warning("No hay datos grabados.")

def fetch_data():   # Function to fetch and clean data from Firebase
    bpm = []
    response = requests.get(firebase_url)
    if response.status_code == 200:
        # Remove quotes and split the data by semicolon
        raw_data = response.text.replace('"', '').strip().split(";")
        # Extract the first element from each group of numbers
        first_elements = []
        secnd_elements = []
        third_elements = []
        for group in raw_data:
            numbers = group.strip().split(" ")  # Split by spaces
            if len(numbers) >= 1:  # Ensure there's at least one number
                try:
                    first_elements.append(float(numbers[0]))
                    secnd_elements.append(float(numbers[1]))
                    third_elements.append(float(numbers[2]))
                    bpm.append(float(numbers[3]))
                except ValueError:
                    continue  # Skip malformed data
        return first_elements,secnd_elements,third_elements,bpm
    else:
        st.error(f"Failed to fetch data from Firebase. Status code: {response.status_code}")
        return []
    
def detect_valid_peaks(ecg_segment, time, threshold):

    pks, _ = find_peaks(ecg_segment)                            # Encontrar los picos en la señal
    valid_pks = ecg_segment[pks][ecg_segment[pks] > threshold]  # Filtrar picos mayores al umbral
    valid_locs = pks[ecg_segment[pks] > threshold]
    peak_times = time[valid_locs] # Obtener los tiempos correspondientes a los picos válidos
    return valid_pks, peak_times

def calculate_features(a):

    arr = np.array(a) / 1000    # Convertir a segundos
    heart_rate = 60 / arr       # Calcular frecuencia cardíaca

    std_hr = np.std(heart_rate)      # Desviación estándar de la frecuencia cardíaca
    max_hr = np.max(heart_rate)      # Frecuencia cardíaca máxima
    median_nni = np.median(a)        # Mediana de los intervalos RR

    triangular_index = get_geometrical_features(a)["triangular_index"]  # Característica geométrica: índice triangular
    rr_differences = np.abs(np.diff(a))                                 # Proporción de diferencias de RR > 20ms (pNNI 20)
    count_differences_20 = np.sum(rr_differences > 20)
    pnni_20 = (count_differences_20 / len(rr_differences)) * 100
    range_nni = np.max(a) - np.min(a)                                   # Rango de intervalos RR
    cvnni = get_time_domain_features(a)["cvnni"]                        # Coeficiente de variación de los intervalos RR
    freq_features = get_frequency_domain_features(a)                    # Características de dominio frecuencial
    hf = freq_features['hf']
    lf_hf_ratio = freq_features['lf_hf_ratio']
    vlf = freq_features['vlf']

    features = {                                                        # Retornar características como un diccionario
        "std_hr": std_hr,
        "max_hr": max_hr,
        "median_nni": median_nni,
        "triangular_index": triangular_index,
        "pnni_20": pnni_20,
        "range_nni": range_nni,
        "cvnni": cvnni,
        "hf": hf,
        "lf_hf_ratio": lf_hf_ratio,
        "vlf": vlf
    }

    return features

def set_background(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """, unsafe_allow_html=True)

def main_page():
    file_id = "190ygOr_J1pqLwvTAXkHMDNyd7Qi5rjS1"
    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    response = requests.get(url)
    set_background("https://img.freepik.com/free-vector/white-abstract-wallpaper_23-2148830027.jpg?t=st=1733431935~exp=1733435535~hmac=7a1d6c26a9eaf01430a8158fd41328553c361e432a078ca188b6ec9f7952d15f&w=996")
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&family=Open+Sans:ital,wght@0,300..800;1,300..800&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet');
        @import url('https://fonts.googleapis.com/css2?family=Bree+Serif&family=Comic+Neue:ital,wght@0,300;0,400;0,700;1,300;1,400;1,700&family=Open+Sans:ital,wght@0,300..800;1,300..800&family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet');
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap" rel="stylesheet');
        .title {
            font-size: 75px;
            font-family: 'Poppins', sans-serif; 
            color: #FD5901;
            text-align: left;
            font-weight: 600; 
            margin-bottom: 5px;
        }
        .subtitle {
            font-size: 40px; 
            color: #000000; 
            text-align: left;
            margin-top: 0px;
            font-family: "Bree Serif", serif;
            font-weight: 400;
            font-style: normal;
        }
        .text-normal {
            font-size: 20px;
            font-family: 'Montserrat', sans-serif; 
            color: #000000; 
            text-align: justify;
            line-height: 1.6;
            margin-top: 10px;
            margin-bottom: 20px;
            font-weight: 300;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="title">CardioArc</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Dispositivo portátil ECG de 3 derivaciones con Inteligencia Artificial</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<p class="text-normal">Este dispositivo utiliza algoritmos avanzados de inteligencia artificial para detectar arritmias cardíacas. Su diseño portátil permite un diagnóstico rápido y preciso en diferentes entornos clínicos, mejorando la calidad del cuidado del paciente.</p>', unsafe_allow_html=True)
    with col2: 
        st.image(response.content,width=300)
    

def visualize_page():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0f1314
        }
        </style>
        """, unsafe_allow_html=True)
    # Inicializar variables de estado para la gráfica
    if "x_data" not in st.session_state:
        st.session_state["x_data"] = []
    if "y_data" not in st.session_state:
        st.session_state["y_data"] = []
    if "global_index" not in st.session_state:
        st.session_state["global_index"] = 1

    # Controles de grabación en el sidebar
    st.sidebar.title("Controles de Grabación")
    if st.sidebar.button("Iniciar Grabación"):
        iniciar_grabacion()
    if st.sidebar.button("Finalizar Grabación"):
        finalizar_grabacion()

    # Configuración de la gráfica
    window_size = 400
    derivada_opcion = st.selectbox(
        "Seleccione la derivada",
        options=["Primera Derivada", "Segunda Derivada", "Tercera Derivada"],
        index=0
    )
    plot_area = st.empty()  # Placeholder para la gráfica

    # Bucle para actualizar la gráfica en tiempo real
    while True:
        # Obtener datos de Firebase
        if derivada_opcion == "Primera Derivada":
            new_y_data, _, _, bpm = fetch_data()
        elif derivada_opcion == "Segunda Derivada":
            _, new_y_data, _, bpm = fetch_data()
        elif derivada_opcion == "Tercera Derivada":
            _, _, new_y_data, bpm = fetch_data()

        bpm = int(np.mean(bpm))
        if new_y_data:
            for value in new_y_data:
                # Agregar datos a las variables de sesión
                st.session_state["x_data"].append(st.session_state["global_index"])
                st.session_state["y_data"].append(value)
                st.session_state["global_index"] += 1

                # Mantener la ventana deslizante
                if len(st.session_state["x_data"]) > window_size:
                    st.session_state["x_data"] = st.session_state["x_data"][-window_size:]
                    st.session_state["y_data"] = st.session_state["y_data"][-window_size:]

                # Crear y actualizar la gráfica
                fig = go.Figure(
                    data=[go.Scatter(x=st.session_state["x_data"], y=st.session_state["y_data"], mode="lines", name="Real-Time Data",line=dict(color="#39FF14", width=2))]
                )
                fig.update_layout(
                    plot_bgcolor = "#000000",paper_bgcolor = "#000000",
                    xaxis=dict(range=[st.session_state["x_data"][0], st.session_state["x_data"][-1]], title="Point Index"),
                    yaxis=dict(range=[min(st.session_state["y_data"]) - 0.5, max(st.session_state["y_data"]) + 0.5], title="Value"),
                    title=f"{derivada_opcion} de su EKG | BPM: {bpm}",
                )
                plot_area.plotly_chart(fig, use_container_width=True)
                time.sleep(0.0035)

                # Guardar datos si la grabación está activa
                if st.session_state["grabando"]:
                    st.session_state["datos_grabados"].append(f"{st.session_state['global_index']}, {value}, BPM: {bpm}")

def analize_page():
    st.title("Analize su ECG")

    # Subir archivo .txt
    uploaded_file = st.file_uploader("Sube tu archivo .txt", type="txt")

    if uploaded_file is not None:
        # Leer las líneas del archivo
        file_content = uploaded_file.read().decode("utf-8")  # Decodificar el archivo como texto
        lines = file_content.strip().split("\n")  # Dividir el contenido en líneas

        # Calcular el número de filas
        num_rows = len(lines)

        # Frecuencia de muestreo
        sampling_frequency = 300  # Hz

        # Calcular el tiempo total de grabación
        total_time_seconds = num_rows / sampling_frequency

        # Mostrar el resultado
        st.write(f"El archivo contiene **{num_rows} muestras**.")
        st.write(f"Frecuencia de muestreo: **{sampling_frequency} Hz**.")
        st.write(f"Tiempo total de grabación: **{total_time_seconds:.2f} segundos**.")

        # Procesar el archivo para obtener la segunda columna (muestras)
        samples = []
        for line in lines:
            parts = line.split(", ")  # Dividir cada línea por comas
            if len(parts) > 1:  # Asegurarse de que haya al menos dos columnas
                try:
                    samples.append(float(parts[1].strip()))  # Segunda columna (muestra)
                except ValueError:
                    continue  # Ignorar errores de formato

        # Generar el eje X (tiempo en segundos)
        
        time_axis = [i / sampling_frequency for i in range(num_rows)]
        ecg_data = np.array(samples)
        time = np.array(time_axis)
        # Crear el gráfico con matplotlib
        plt.figure(figsize=(10, 5))
        plt.plot(time, ecg_data, label="Muestras")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Valor de la Muestra")
        plt.title("Gráfico ECG")
        plt.grid()
        plt.legend()
        st.pyplot(plt)


        valid_pks_ecg, peak_times_ecg = detect_valid_peaks(ecg_data, time, 2.7)
        plt.figure(figsize=(10, 5))
        plt.plot(time, ecg_data, label='ECG data')
        plt.plot(peak_times_ecg, valid_pks_ecg,'ro', label='Picos > 0.5', markerfacecolor='red')
        plt.title('Picos detectados en ECG')
        plt.grid()
        st.pyplot(plt)

        rr_interval = np.diff(peak_times_ecg)*1000
        st.write("Intervalos entre Pico R-R (ms):", rr_interval)

        features = calculate_features(rr_interval)

        std_hr = features['std_hr']
        max_hr = features['max_hr']
        median_nni = features['median_nni']
        triangular_index = features['triangular_index']
        pnni_20 = features['pnni_20']
        range_nni = features['range_nni']
        cvnni = features['cvnni']
        hf = features['hf']
        lf_hf_ratio = features['lf_hf_ratio']
        vlf = features['vlf']

        
        st.write(f"Desviación estándar del ritmo cardíaco: {features['std_hr']:.2f} ms")
        st.write(f"Frecuencia cardíaca máxima: {features['max_hr']:.2f} bpm")
        st.write(f"Mediana de los intervalos NN: {features['median_nni']:.2f} ms")
        st.write(f"Índice triangular: {features['triangular_index']:.2f}")
        st.write(f"Proporción de diferencias de RR > 20ms (pNNI 20): {features['pnni_20']:.2f}%")
        st.write(f"Rango de intervalos NN: {features['range_nni']:.2f} ms")
        st.write(f"Coeficiente de variación de los intervalos RR (CVNNI): {features['cvnni']:.2f}")
        st.write(f"Energía en la banda HF: {features['hf']:.2f} ms²")
        st.write(f"Relación LF/HF: {features['lf_hf_ratio']:.2f}")
        st.write(f"Energía en la banda VLF: {features['vlf']:.2f} ms²")

        columns = ["std_hr", "max_hr", "median_nni", "triangular_index", "pnni_20", "range_nni", "cvnni", "hf", "lf_hf_ratio", "vlf"]
        data = np.array([[std_hr, max_hr, median_nni, triangular_index, pnni_20, range_nni, cvnni, hf, lf_hf_ratio, vlf]])
        df = pd.DataFrame(data, columns=columns)
        y_predicted = model.predict(df)
        st.write(y_predicted)



def login_page():
    st.title("Iniciar Sesión")
    st.markdown("Por favor, introduce tus credenciales para continuar.")

    username = st.text_input("Usuario", placeholder="Introduce tu usuario")
    password = st.text_input("Contraseña", type="password", placeholder="Introduce tu contraseña")

    if st.button("Iniciar Sesión"):
        if username in st.session_state["users"] and st.session_state["users"][username] == password:
            st.session_state["authentication_status"] = True
            st.success(f"¡Bienvenido de nuevo, {username}!")
            st.session_state["page"] = "Inicio"
        else:
            st.session_state["authentication_status"] = False
            st.error("Usuario o contraseña incorrectos.")

def signup_page():
    st.title("Registrarse")
    st.markdown("Crea una cuenta para utilizar la aplicación.")

    new_username = st.text_input("Nuevo Usuario", placeholder="Elige un nombre de usuario")
    new_password = st.text_input("Nueva Contraseña", type="password", placeholder="Elige una contraseña")

    if st.button("Registrar"):
        if new_username in st.session_state["users"]:
            st.error("Este usuario ya existe. Por favor, elige otro.")
        elif len(new_password) < 6:
            st.error("La contraseña debe tener al menos 6 caracteres.")
        else:
            st.session_state["users"][new_username] = new_password
            st.success("¡Usuario registrado con éxito! Ahora puedes iniciar sesión.")
            st.session_state["page"] = "Login"

# Inicializar variables de sesion
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "users" not in st.session_state:
    st.session_state["users"] = {"admin": "admin123"}  # Usuario por defecto
if "page" not in st.session_state:
    st.session_state["page"] = "Inicio"  # Página predeterminada
if "grabando" not in st.session_state:
    st.session_state["grabando"] = False
if "datos_grabados" not in st.session_state:
    st.session_state["datos_grabados"] = []

selected = option_menu(
    menu_title=None,
    options=["Inicio","Visualize su ECG", "Analize su ECG", "Registro", "Inicia Sesión"],
    icons=["house","heart-pulse", "clipboard2-pulse", "person-add", "person-check"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0.5rem", "background-color": "#005F60","width": "100%"},
        "nav-link-selected": {"background-color": "#fb8b24", "color": "#000000"},  
    },
)
# Actualización de la página actual basada en selección
if selected == "Inicio":
    st.session_state["page"] = "Inicio"
elif selected == "Visualize su ECG":
    st.session_state["page"] = "visualize"
elif selected == "Analize su ECG":
    st.session_state["page"] = "analize"
elif selected == "Registro":
    st.session_state["page"] = "Signup"
elif selected == "Inicio Sesión":
    st.session_state["page"] = "Login"

# Renderizado dinámico de páginas
if st.session_state["page"] == "Inicio":
    main_page()
elif st.session_state["page"] == "visualize":
    st.empty()
    visualize_page()
elif st.session_state["page"] == "analize":
    analize_page()
elif st.session_state["page"] == "Signup":
    signup_page()
elif st.session_state["page"] == "Login":
    login_page()
