import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft
from socketio import Client
import time
import mysql.connector
import datetime
from io import BytesIO

# Parameters
sampling_rate = 48000  # Hz
fft_size = 8192  # 2^13
overlap = 0.5  # 50% overlap
window_type = 'hamming'  # Hamming window

# Database connection setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Shri_0310",
    database="eol_db",
    auth_plugin="mysql_native_password"
)
cur = conn.cursor()

# SocketIO Client
sio = Client()

try:
    sio.connect('http://127.0.0.1:5000')
except Exception as e:
    print(f'Connection failed: {e}')

progress_data = []

# Function to store graph in MySQL

# Function to store the graph as binary in MySQL
def store_graph_in_db(graph_type, file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    
    query = "INSERT INTO graphs (graph_type, image) VALUES (%s, %s)"
    cur.execute(query, (graph_type, binary_data))
    conn.commit()

# Function to read ASCII data
def read_ascii_data(file_path):
    data = np.loadtxt(file_path)
    return data

# Function to perform FFT and calculate amplitude spectrum
def calculate_fft(data, sampling_rate, window_type, overlap, fft_size):
    window = get_window(window_type, fft_size)
    step = int(fft_size * (1 - overlap))
    freqs = np.fft.rfftfreq(fft_size, 1 / sampling_rate)
    fft_results = []
    num_channels = data.shape[1] if data.ndim > 1 else 1
    data = data.reshape(-1, num_channels)

    total_steps = (len(data) - fft_size) // step + 1

    for step_num, start in enumerate(range(0, len(data) - fft_size + 1, step)):
        segment = data[start:start + fft_size, :] * window[:, np.newaxis]
        fft_result = np.abs(fft(segment, axis=0)[:fft_size // 2 + 1, :])
        fft_results.append(fft_result)

        # Update progress
        progress = (step_num + 1) / total_steps * 100
        timestamp = datetime.datetime.now()
        progress_data.append((timestamp, progress))        
        try:
            sio.emit('update_progress', {'progress': progress})
        except Exception as e:
            print(f'Error sending progress: {e}')

    fft_results = np.array(fft_results)
    amplitude = np.abs(fft_results) / len(fft_results)
    peak_value = np.max(np.abs(fft_results), axis=0)

    return freqs, fft_results, amplitude, peak_value
# Define true value and set acceptable limits
true_value = 0.1  # Example true value
lower_limit = true_value * 0.97  # 3% below true value
upper_limit = true_value * 1.03  # 3% above true value

# Function to average FFT results
def average_fft_results(fft_results):
    return np.mean(fft_results, axis=0)

# Function to calculate RMS value
def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2, axis=0))

# Check if RMS values are within the limit
def check_rms_within_limit(rms_value, lower_limit, upper_limit):
    if np.any(rms_value < lower_limit) or np.any(rms_value > upper_limit):
        return 'Bad Sample'
    else:
        return 'Good Sample'

# Main processing function
def main(file_path):
    # Read raw data
    raw_data = read_ascii_data(file_path)

    # Perform FFT
    freqs, fft_results, amplitude, peak_value = calculate_fft(raw_data, sampling_rate, window_type, overlap, fft_size)

    # Average FFT results
    averaged_fft = average_fft_results(fft_results)

    # Calculate RMS value over time
    segment_size = int(fft_size * (1 - overlap))
    rms_values = [calculate_rms(raw_data[i:i + segment_size]) for i in range(0, len(raw_data) - segment_size + 1, segment_size)]
    rms_values = np.array(rms_values)
    times = np.arange(len(rms_values)) * segment_size / sampling_rate

    # Calculate overall RMS value
    overall_rms_value = calculate_rms(raw_data)
    rounded_rms_value = np.round(overall_rms_value, 2)

    # Check if RMS value is within the limit
    sample_status = check_rms_within_limit(overall_rms_value, lower_limit, upper_limit)

    # Emit results
    try:
        sio.emit('update_results', {
            'sample_status': sample_status,
            'overall_rms_value': rounded_rms_value.tolist(),
            'upper_limit': upper_limit,
            'lower_limit': lower_limit
        })
    except Exception as e:
        print(f'Error sending results: {e}')
    
     # Plot average frequency domain signal (Frequency vs Amplitude)
    plt.figure(figsize=(10, 8))
    plt.plot(freqs[:fft_size // 2], averaged_fft[:fft_size // 2])
    plt.xlabel('Frequency (Hz)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    plt.title('Frequency Spectrum', fontsize=18)
    fft_image_path = 'static/fft_graphs.png'
    plt.savefig(fft_image_path)
    plt.close()

    # Store the FFT graph in the database
    store_graph_in_db('Frequency Spectrum', fft_image_path)

    # Plot RMS value over time
    plt.figure()
    plt.plot(times, rms_values)
    plt.axhline(y=upper_limit, color='g', linestyle='--', label='Upper Limit')
    plt.axhline(y=lower_limit, color='r', linestyle='--', label='Lower Limit')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Value')
    plt.title('Level Analysis')
    plt.legend()
    rms_image_path = 'static/rms_graphs.png'
    plt.savefig(rms_image_path)
    plt.close()

    # Store the RMS graph in the database
    store_graph_in_db('Level Analysis', rms_image_path)

# Execute the main function
main('E:\\EOL_testing\\raw_data.asc')
