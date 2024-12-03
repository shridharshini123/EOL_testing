import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.fft import fft
from socketio import Client
import time
import mysql.connector
import datetime

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
    sio.connect('http://127.0.0.1:5000')  # Corrected URL without extra path
except Exception as e:
    print(f'Connection failed: {e}')  # Debugging line
progress_data = []

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

# Function to average FFT results
def average_fft_results(fft_results):
    return np.mean(fft_results, axis=0)

# Function to calculate RMS value
def calculate_rms(data):
    return np.sqrt(np.mean(data ** 2, axis=0))

# Define true value and set acceptable limits
true_value = 0.1  # Example true value
lower_limit = true_value * 0.97  # 3% below true value
upper_limit = true_value * 1.03  # 3% above true value

# Check if RMS values are within the limit
def check_rms_within_limit(rms_value, lower_limit, upper_limit):
    if np.any(rms_value < lower_limit) or np.any(rms_value > upper_limit):
        return 'Bad Sample'
    else:
        return 'Good Sample'
# Function to plot progress data
def plot_progress():
    if not progress_data:
        print("No progress data to plot.")
        return

    timestamps, progress = zip(*progress_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, progress)
    plt.xlabel('Timestamp')
    plt.ylabel('Progress (%)')
    plt.title('Algorithm Progress Over Time')
    plt.savefig('static/progress_graph.png')
    plt.close()

def insert_fft_analysis_data(product_id, serial_no, timestamp, sampling_rate, bandwidth, spectrum_lines, spectrum_sizes, overlap, windowing, averaging_type):
    insert_query = """
    INSERT INTO FFT_analysis_data (product_id, serial_no, timestamp, sampling_rate, bandwidth, spectrum_lines, spectrum_sizes, overlap, windowing, averaging_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.execute(insert_query, (product_id, serial_no, timestamp, sampling_rate, bandwidth, spectrum_lines, spectrum_sizes, overlap, windowing, averaging_type))
    conn.commit()

def insert_rms_value(signal_id, sensor_id, product_id, frequency, rms_value, peak_value, amplitude, recorded_at, serial_no):
    # Convert NumPy arrays to Python lists
    frequency_list = frequency.tolist()
    amplitude_list = amplitude.tolist()
    rms_value_list = rms_value.tolist()
    
    # Flatten lists into strings for storage
    frequency_str = ','.join(map(str, frequency_list))
    amplitude_str = ','.join(map(str, amplitude_list))
    rms_value_str = ','.join(map(str, rms_value_list))

    # Ensure rms_value and peak_value are scalars
    if isinstance(rms_value, np.ndarray):
        rms_value = float(np.mean(rms_value))  # Take the mean as a representative value
    if isinstance(peak_value, np.ndarray):
        peak_value = float(np.max(peak_value))  # Take the max as a representative value

    insert_query = """
    INSERT INTO vibration_signal (signal_id, sensor_id, product_id, frequency, rms_value, peak_value, amplitude, recorded_at, serial_no)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.execute(insert_query, (signal_id, sensor_id, product_id, frequency_str, rms_value_str, peak_value, amplitude_str, recorded_at, serial_no))
    conn.commit()

def insert_status(analysis_id, product_id, analysis_date, analysis_result, serial_no):
    insert_query = """
    INSERT INTO status (analysis_id, product_id, analysis_date, analysis_result, serial_no)
    VALUES (%s, %s, %s, %s, %s)
    """
    cur.execute(insert_query, (analysis_id, product_id, analysis_date, analysis_result, serial_no))
    conn.commit()

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
    rounded_rms_value = np.round(overall_rms_value, 2)  # Round each element to 2 decimal places

    print(overall_rms_value)
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
        print(f'Error sending results: {e}')  # Debugging line
    # Plot average frequency domain signal (Frequency vs Amplitude)

    plt.figure(figsize=(10, 8))
    plt.plot(freqs[:fft_size // 2], averaged_fft[:fft_size // 2])
    plt.xlabel('Frequency (Hz)', fontsize=16)  # Increase label font size
    plt.ylabel('Amplitude', fontsize=16)  # Increase label font size
    plt.title('Frequency Spectrum', fontsize=18)  # Increase title font size
    plt.savefig('static/fft_graphs.png')
    plt.show()
    plt.close()

    # Plot RMS value over time
    plt.plot(times, rms_values)
    plt.axhline(y=upper_limit, color='g', linestyle='--', label='Upper Limit')
    plt.axhline(y=lower_limit, color='r', linestyle='--', label='Lower Limit')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Value')
    plt.title('Level Analysis')
    plt.legend()
    plt.savefig('static/rms_graphs.png')
    plt.show()
    plt.close()

    # Plot progress
    plot_progress()

    
    product_id = "NSP001"  # Replace with actual product ID
    # Insert FFT analysis data into the database
    timestamp = datetime.datetime.now()
    bandwidth = sampling_rate / 2
    spectrum_lines = fft_size // 2
    spectrum_sizes = fft_size
    windowing = 1  # Assuming 1 for Hamming window
    averaging_type = 1  # Assuming 1 for average type
    insert_fft_analysis_data(
        product_id=product_id,
        serial_no="SN123456",
        timestamp=timestamp,
        sampling_rate=sampling_rate,
        bandwidth=bandwidth,
        spectrum_lines=spectrum_lines,
        spectrum_sizes=spectrum_sizes,
        overlap=overlap,
        windowing=windowing,
        averaging_type=averaging_type
    )

    # Insert RMS value into the database
    insert_rms_value(
        signal_id="SIG12345",
        sensor_id="SENS001",
        product_id=product_id,
        frequency=freqs,
        rms_value=rounded_rms_value,
        peak_value=peak_value,
        amplitude=amplitude,
        recorded_at=timestamp,
        serial_no="SN123456"
    )

    # Insert status into the database
    insert_status(
        analysis_id="ANAL1234",
        product_id=product_id,
        analysis_date=timestamp,
        analysis_result=sample_status,
        serial_no="SN123456"
    )

    
       

if __name__ == "__main__":
    file_path = "E:\\CyberVault\\Project\\SQL\\raw_data.asc"  # Replace with your file path
    main(file_path)
