import mysql.connector
import queue
from threading import Thread, Lock
import time
from flask import Flask, render_template, request, session, redirect, url_for, flash, send_file, abort,make_response
import io
from datetime import datetime  # Importing datetime directly
import re
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from flask_session import Session
import os
import base64
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'  # or 'redis', 'memcached', etc.
Session(app)    
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Establish database connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Shri_0310",
    auth_plugin="mysql_native_password"
)

cur = conn.cursor()
# Global variables to store progress and results
current_progress = {'progress': 0}
current_results = {
    'sample_status': '',
    'overall_rms_value': [],
    'upper_limit': 0,
    'lower_limit': 0
}
# Create and use database
cur.execute("CREATE DATABASE IF NOT EXISTS EOL_DB;")
cur.execute("USE EOL_DB;")

# Create tables
cur.execute('''CREATE TABLE IF NOT EXISTS item(
                product_id VARCHAR(50) PRIMARY KEY,
                product_name VARCHAR(50),
                product_type VARCHAR(50),
                part_no VARCHAR(50),
                customer_no VARCHAR(50)
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS sensor(
                sensor_id INT AUTO_INCREMENT PRIMARY KEY,
                sensor_type VARCHAR(50),
                manufacturer VARCHAR(50),
                model VARCHAR(50),
                location VARCHAR(50),
                installation_date DATE,
                calibration_date DATE,
                status VARCHAR(30) DEFAULT 'active',
                sensitivity_value INT
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS vibration_signal (
                signal_id INT AUTO_INCREMENT PRIMARY KEY,
                sensor_id INT,
                product_id VARCHAR(50),
                frequency DECIMAL(10,2),
                rms_value DECIMAL(10,2),
                peak_value DECIMAL(10,2),
                amplitude DECIMAL(10,2),
                recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sensor_id) REFERENCES sensor(sensor_id),
                FOREIGN KEY (product_id) REFERENCES item(product_id)
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS status(
                analysis_id INT AUTO_INCREMENT PRIMARY KEY,
                product_id VARCHAR(50),
                analyst_id INT,
                analysis_date DATE,
                analysis_result VARCHAR(50),
                inspection_details TEXT,
                FOREIGN KEY (product_id) REFERENCES item(product_id)
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS FFT_analysis_data(
                product_id VARCHAR(50),
                timestamp DATETIME NOT NULL,
                sampling_rate FLOAT,
                bandwidth FLOAT,
                spectrum_lines FLOAT,
                spectrum_sizes FLOAT,
                overlap FLOAT,
                windowing FLOAT,
                averaging_type FLOAT,
                FOREIGN KEY (product_id) REFERENCES item(product_id)
            )''')

cur.execute('''    CREATE TABLE IF NOT EXISTS raw_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        file_path VARCHAR(50),
        time TIME,
        date DATE
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS defect_types (
                defect_id INT AUTO_INCREMENT PRIMARY KEY,
                product_id VARCHAR(50),
                defeat_status VARCHAR(50),
                FOREIGN KEY (product_id) REFERENCES item(product_id)
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS reference(
                sensor_id INT,
                product_id VARCHAR(50),
                sensor_type VARCHAR(50),
                freq_lower_limit VARCHAR(20),
                freq_upper_limit VARCHAR(20),
                threshold_upper_limit INT,
                threshold_lower_limit INT,
                sensitivity INT,
                serial_no VARCHAR(50),
                signal_to_noise INT,
                FOREIGN KEY (sensor_id) REFERENCES sensor(sensor_id)
            )''')

cur.execute('''CREATE TABLE IF NOT EXISTS testing (
    test_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id VARCHAR(50),
    test_date DATE,
    test_time TIME,
    test_duration TIME,
    test_result VARCHAR(50),
    FOREIGN KEY (product_id) REFERENCES item(product_id)
)''')

cur.execute('''CREATE TABLE IF NOT EXISTS user(
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            emp_id VARCHAR(50),
            name VARCHAR(50),
            email VARCHAR(100),
            mobile VARCHAR(50),
            password VARCHAR(50),
            designation VARCHAR(50),
            profile_image BLOB,
            role VARCHAR(50))''')
# Queue and threading
fifo_queue = queue.Queue()
lock = Lock()

# Check if connection is successful
if conn.is_connected():
    print("Connected to MySQL database")
# Function to insert file location into raw_data table
# Function to insert file location into raw_data table
def insert_file_location(file_location, status):
    try:
        # Get the current time and date
        current_time = datetime.now().time()
        current_date = datetime.now().date()

        # Insert the file location with time, date, and status
        cur.execute(
            "INSERT INTO raw_data (file_path, time, date, status) VALUES (%s, %s, %s, %s)",
            (file_location, current_time, current_date, status)
        )
        conn.commit()
        return True  # Return True if successful
    except Exception as e:
        print(f"Error during insertion: {e}")
        conn.rollback()
        return False  # Return False if there's an error

# Function to process the queue
def process_queue():
    while True:
        try:
            with lock:
                if not fifo_queue.empty():
                    sql_item = fifo_queue.get()
                    if isinstance(sql_item, tuple) and len(sql_item) == 2:
                        sql, values = sql_item
                        cur.execute(sql, values)
                        conn.commit()
        except Exception as e:
            print(f"Error executing SQL: {e}")
            conn.rollback()  # Rollback in case of error
        finally:
            time.sleep(0.1)

# Start the queue processing thread
fifo_thread = Thread(target=process_queue)
fifo_thread.start()
# Function to validate password complexity
def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[!@#$%^&*(),.?\"_:{}|<>]", password):
        return False, "Password must contain at least one special character (!@#$%^&*(),.?\"_:{}|<>)."
    return True, None

def fetch_file_path_from_db(file_id):
    # Implement your database fetching logic here
    # For example:
    # cursor.execute("SELECT file_path FROM files WHERE id = %s", (file_id,))
    # return cursor.fetchone()[0]
    return r"E:\\CyberVault\\Project\\SQL\\raw_data.asc"  # Replace with your actual query result
# Base64 filter for encoding binary data
@app.template_filter('b64encode')
def b64encode_filter(data):
    if data:
        return base64.b64encode(data).decode('utf-8')
    return ''
app.jinja_env.filters['b64encode'] = b64encode_filter

@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']

    # Fetch user's name and profile image using the email
    cur.execute("SELECT name, profile_image FROM user WHERE email = %s", (email,))
    user = cur.fetchone()
    
    if user:
        name = user[0]
        profile_image = user[1]
    else:
        name = "User"
        profile_image = None

    # Store user data in session
    session['name'] = name
    session['profile_image'] = profile_image

    return render_template('home.html', name=name, profile_image=profile_image)
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        cur = conn.cursor()

        # Example SQL query to check credential s (sanitize inputs to prevent SQL injection)
        cur.execute("SELECT email, password FROM user WHERE email=%s AND password=%s", (email, password))
        user = cur.fetchall()

        if user:
            # Store user email in session
            session['email'] = email
            return redirect(url_for('home'))  # Redirect to home page on successful login
        else:
            error_message = "Invalid email and password."
            return render_template('login.html', error=error_message)
    return render_template('login.html')  # For GET requests or initial load of the page
@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']

    if request.method == 'POST':
        mobile = request.form['mobile']
        des = request.form['des']
        role = request.form['role']
        address = request.form['address']
        profile_image = request.files['profile_image']

        if profile_image and profile_image.filename != '':
            profile_image_data = profile_image.read()
        else:
            profile_image_data = None

        if profile_image_data:
            sql = '''UPDATE user SET mobile = %s, designation = %s, role = %s, address = %s, profile_image = %s WHERE email = %s'''
            values = (mobile, des, role, address, profile_image_data, email)
        else:
            sql = '''UPDATE user SET mobile = %s, designation = %s, role = %s, address = %s WHERE email = %s'''
            values = (mobile, des, role, address, email)

        cur.execute(sql, values)
        conn.commit()

        session['mobile'] = mobile
        session['designation'] = des
        session['role'] = role
        session['address'] = address
        if profile_image_data:
            session['profile_image'] = profile_image_data

        flash('Profile updated successfully.', 'success')
        return redirect(url_for('home'))

    cur.execute("SELECT name, mobile, designation, role, profile_image, address FROM user WHERE email=%s", (email,))
    user_data = cur.fetchone()

    user = {
        'name': user_data[0] if user_data else '',
        'email': email,
        'mobile': user_data[1] if user_data else '',
        'designation': user_data[2] if user_data else '',
        'role': user_data[3] if user_data else '',
        'profile_image': user_data[4] if user_data else None,
        'address': user_data[5] if user_data else ''
    }

    return render_template('profile.html', user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        user = request.form['user']
        em = request.form['email']
        pwd = request.form['pwd']
        cpwd = request.form['cpwd']

        # Check if the email already exists in the database
        cur.execute("SELECT email FROM user WHERE email=%s", (em,))
        existing_user = cur.fetchone()
        
        if existing_user:
            msg = "Email already registered. Please use a different email."
            return render_template('sign_up.html', error=msg)
        
        # Validate password
        is_valid, error_message = validate_password(pwd)
        if not is_valid:
            return render_template('sign_up.html', error=error_message)
        
        if pwd == cpwd:
            # Fetch the next available emp_id
            cur.execute("SELECT MAX(user_id) FROM user")
            max_id = cur.fetchone()[0]
            emp_id = f"EMP{max_id + 1:04d}" if max_id is not None else "EMP0001"
            cur.execute("INSERT INTO user(emp_id, name, email, password) VALUES(%s, %s, %s, %s)", (emp_id, user, em, pwd))
            conn.commit()
            return redirect(url_for('login'))  # Redirect to login page after successful signup
        else:
            msg = "Passwords do not match."
            return render_template('sign_up.html', error=msg)
    
    return render_template('sign_up.html')
@app.route('/forgot', methods=['GET', 'POST'])
def forgot():
    if request.method == "POST":
        email = request.form['email']
        new_password = request.form['pwd']
        confirm_password = request.form['re_pwd']

        # Validate the email
        cur = conn.cursor()
        cur.execute("SELECT email FROM user WHERE email=%s", (email,))
        user = cur.fetchone()

        if not user:
            flash('Email not found.', 'error')
            return render_template('forgot.html')

        # Validate the new password
        is_valid, error_message = validate_password(new_password)
        if not is_valid:
            flash(error_message, 'error')
            return render_template('forgot.html')

        if new_password != confirm_password:
            flash('New passwords do not match.', 'error')
            return render_template('forgot.html')
        
        # Hash the new password

        # Update password in the database
        cur.execute("UPDATE user SET password=%s WHERE email=%s", (new_password, email))
        conn.commit()

        flash('Password updated successfully. Please log in with your new password.', 'success')
        return redirect(url_for('login'))

    return render_template('forgot.html')
@app.route('/logout',methods=['GET','POST'])
def logout():
        # Clear the session
    session.pop('user_id', None)
    session.pop('username', None)
    
    # Redirect to the login page
    return redirect(url_for('login'))

@app.route('/sample_details', methods=['GET', 'POST'])
def item_report():

    if request.method == "GET":
        cur = conn.cursor()

        cur.execute("SELECT i.product_id,t.test_id, i.product_name, i.product_type, i.part_no, i.customer_no,i.serial_no  FROM item i JOIN testing t ON i.product_id = t.product_id")
        items = cur.fetchall()  # Fetch all rows from the joined tables
        item_headings = ["Product ID", "Test ID", "Product Name", "Product Type", "Part No", "Customer No","Serial No"]

        if items:
            lenn = str(100 / len(items[0])) + "%"  # Calculate width per column
        else:
            lenn = "100%"  # Default width if no items
        cur.close()

        return render_template('details.html', item_headings=item_headings, icon=items)
'''
@app.route('/raw_data', methods=['GET', 'POST'])
def raw_data():
    cur = conn.cursor()
    message = ""  # Initialize the message variable

    if request.method == "POST":
        # Handle file location insertion
        file_location = request.form.get('file_location')
        if file_location:
            file_location = file_location.strip()  # Remove any leading/trailing whitespace
            
            # Fetch the total number of entries in raw_data
            cur.execute("SELECT COUNT(*) FROM raw_data")
            row_count = cur.fetchone()[0]

            # Fetch the corresponding product_id from the item table
            # Using OFFSET to get the product_id for the current row count
            cur.execute("SELECT product_id FROM item LIMIT 1 OFFSET %s", (row_count,))
            product_id_result = cur.fetchone()

            if product_id_result:
                product_id = product_id_result[0]

                # Insert into raw_data with the fetched product_id
                cur.execute(
                    "INSERT INTO raw_data (file_path, time, date, status, product_id) VALUES (%s, NOW(), CURDATE(), 'Ongoing', %s)",
                    (file_location, product_id)
                )
                conn.commit()
                message = f"File location '{file_location}' inserted successfully for product ID {product_id}."
            else:
                message = "No more products available in the item table."
        else:
            message = "File location cannot be empty."

        # Handle status update
        status_update = request.form.get('status_update')
        raw_id = request.form.get('raw_id')
        if status_update and raw_id:
            if status_update in ['Ongoing', 'Completed', 'Not started']:
                cur.execute("UPDATE raw_data SET status=%s WHERE id=%s", (status_update, raw_id))
                conn.commit()
                message = f"Status updated to '{status_update}' for ID {raw_id}."
            else:
                message = "Invalid status value."

    # Fetch updated data after insertion or status update
    cur.execute("SELECT id, file_path, time, DATE_FORMAT(date, '%d-%m-%Y') AS formatted_date, status, product_id FROM raw_data")
    items = cur.fetchall()
    item_headings = ["ID", "File Location", "Time", "Date", "Status", "Product ID"]

    cur.close()

    return render_template('raw.html', raw_data=item_headings, con=items, message=message)

'''
@app.route('/raw_data', methods=['GET', 'POST'])
def raw_data():
    try:
        # Fetch the latest entry from the raw_data table
        cur.execute("SELECT file_path, DATE_FORMAT(time, '%H:%i') AS formatted_time, date, product_id FROM raw_data ORDER BY id DESC LIMIT 1")
        latest_raw_data = cur.fetchone()
        print("Latest raw data:", latest_raw_data)

        # Initialize default values
        file_path, formatted_time, file_date, product_id = ('None', 'N/A', 'N/A', None)

        if latest_raw_data:
            file_path, formatted_time, file_date, product_id = latest_raw_data

        # Initialize product_name and serial_no
        product_name, serial_no = 'Unknown Product', 'N/A'

        # Fetch product_name and serial_no from the item table using the product_id
        if product_id:
            cur.execute("SELECT product_name, serial_no FROM item WHERE product_id = %s", (product_id,))
            item_data = cur.fetchone()
            print("Item data:", item_data)
            if item_data:
                product_name, serial_no = item_data

        # Fetch test_id from the testing table for the specific product_id
        cur.execute("SELECT test_id FROM testing WHERE product_id = %s LIMIT 1", (product_id,))
        testing_data = cur.fetchone()
        print("Testing data:", testing_data)
        test_id = testing_data[0] if testing_data else 'N/A'

        # Fetch sample_status and overall_rms_value from the result_data table
        cur.execute("SELECT sample_status, overall_rms_value FROM result_data")
        result_data = cur.fetchone()
        print("Result data:", result_data)

        if result_data:
            sample_status, overall_rms_value = result_data
            overall_rms_value_list = overall_rms_value.split(',') if isinstance(overall_rms_value, str) else []

            # Change 'Bad Sample' to 'Not OK'
            if sample_status == 'Bad Sample':
                sample_status = 'NOK'
        else:
            sample_status = 'N/A'
            overall_rms_value_list = []

        # Count number of 'Bad Sample'
        cur.execute("SELECT COUNT(sample_status) FROM result_data WHERE sample_status='Bad Sample'")
        bad_sample_count = cur.fetchone()[0] or 0
        print("Count of Bad Samples:", bad_sample_count)

        # Count total samples
        cur.execute("SELECT COUNT(sample_status) FROM result_data")
        total_sample_count = cur.fetchone()[0] or 0
        print("Total Samples Count:", total_sample_count)

        # Prepare the res_items string
        res_items = f"{bad_sample_count}/{total_sample_count}" if total_sample_count > 0 else "0/0"
        print("Formatted res_items:", res_items)

        # Check if the file exists and read it
        if file_path and os.path.isfile(file_path):
            time_values = [] 
            amplitude_values = []

            with open(file_path, 'r') as file:
                for line in file:
                    columns = line.split()
                    if len(columns) >= 2:
                        time_values.append(float(columns[0]))  # Time values
                        amplitude_values.append(float(columns[1]))  # Amplitude values

            # Plot the graph
            plt.figure()
            plt.plot(time_values, amplitude_values)
            plt.xlabel('Time')
            plt.ylabel('Amplitude')

            # Convert the plot to an image in memory
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()  # Close the figure
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode('utf8')
        else:
            graph_url = None

        # Pass the fetched data to the template
        return render_template(
            'raw.html',
            product_name=product_name,
            serial_no=serial_no,
            test_id=test_id,
            file_path=file_path,
            sample_status=sample_status,
            overall_value=overall_rms_value_list,
            time=formatted_time,
            date=file_date,
            res_items=res_items,
            graph_url=graph_url,
            graph=img
        )

    except Exception as e:
        print(f'Error fetching data: {e}')
        return render_template('raw.html', error=str(e))

@app.route('/download_file/<file_id>', methods=['GET'])
def download_file(file_id):
    # Fetch the file path from the database based on file_id
    file_path = fetch_file_path_from_db(file_id)

    # Check if the file exists
    if not os.path.isfile(file_path):
        abort(404)  # File not found, raise 404 error

    try:
        # Send the file as a downloadable response
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return str(e), 500  # Handle any other exceptions


@app.route('/delete', methods=['POST'])
def delete():
    raw_id = request.form.get('raw_id')
    
    if not raw_id:
        return "No ID provided for deletion.", 400  # Return 400 for bad request

    max_retries = 3  # Retry the transaction 3 times before failing
    attempt = 0

    while attempt < max_retries:
        try:
            # Perform the delete operation
            cur.execute("DELETE FROM raw_data WHERE id=%s", (raw_id,))
            conn.commit()

            return f"Record with ID {raw_id} deleted successfully.", 200  # Return success message
        
        except mysql.connector.errors.InternalError as e:
            if e.errno == 1213:  # Error code for deadlock
                conn.rollback()  # Rollback the transaction
                attempt += 1
                time.sleep(1)  # Wait before retrying
            else:
                raise  # Reraise other exceptions
    
    return "Failed to delete record due to database deadlock.", 500  # Return error if all retries fail

@app.route('/rms')
def rms():
    #cur.execute("SELECT rms_value FROM vibration_signal")
    #values = cur.fetchall()
    # If you expect only one value, get the first item from the result
    #if values:
     #   rms_value = values[0][0]  # Extract the value from the first tuple
    ##   rms_value = "No data available"
    # Fetch the latest entry from the raw_data table
    cur.execute("SELECT file_path,  DATE_FORMAT(time, '%H:%i') AS formatted_time, date, product_id FROM raw_data ORDER BY id DESC LIMIT 1")
    latest_raw_data = cur.fetchone()
    print("Latest raw data:", latest_raw_data)

    # Initialize default values
    file_path, formatted_time, file_date, product_id = ('None', 'N/A', 'N/A', None)

    if latest_raw_data:
        file_path, formatted_time, file_date, product_id = latest_raw_data

    # Initialize product_name and serial_no
    product_name, serial_no = 'Unknown Product', 'N/A'

    # Fetch product_name and serial_no from the item table using the product_id
    if product_id:
        cur.execute("SELECT product_name, serial_no FROM item WHERE product_id = %s", (product_id,))
        item_data = cur.fetchone()
        print("Item data:", item_data)
        if item_data:
            product_name, serial_no = item_data

    # Fetch test_id from the testing table for the specific product_id
    cur.execute("SELECT test_id FROM testing WHERE product_id = %s LIMIT 1", (product_id,))
    testing_data = cur.fetchone()
    print("Testing data:", testing_data)
    test_id = testing_data[0] if testing_data else 'N/A'

    # Fetch sample_status and overall_rms_value from the result_data table
    cur.execute("SELECT sample_status, overall_rms_value FROM result_data")
    result_data = cur.fetchone()
    print("Result data:", result_data)

    if result_data:
        sample_status, overall_rms_value = result_data
        overall_rms_value_list = overall_rms_value.split(',') if isinstance(overall_rms_value, str) else []

        # Change 'Bad Sample' to 'Not OK'
        if sample_status == 'Bad Sample':
            sample_status = 'NOK'
    else:
        sample_status = 'N/A'
        overall_rms_value_list = []
 # Fetch FFT and RMS Graphs from the 'graphs' table
    cur.execute("SELECT image FROM graphs WHERE graph_type = 'Frequency Spectrum' ORDER BY id DESC LIMIT 1")
    fft_graph = cur.fetchone()

    cur.execute("SELECT image FROM graphs WHERE graph_type = 'Level Analysis' ORDER BY id DESC LIMIT 1")
    rms_graph = cur.fetchone()

    return render_template('rms.html', product_name=product_name,
            serial_no=serial_no,
            test_id=test_id,
            file_path=file_path,
            sample_status=sample_status,
            overall_value=overall_rms_value_list,
            time=formatted_time,
            date=file_date, fft_graph=fft_graph[0], rms_graph=rms_graph[0])
@app.route('/graph/<graph_type>')
def display_graph(graph_type):
    # Fetch the correct graph type
    cur.execute("SELECT image FROM graphs WHERE graph_type = %s ORDER BY id DESC LIMIT 1", (graph_type,))
    result = cur.fetchone()

    if result:
        image_data = result[0]
        # Return the image data as a response
        response = make_response(image_data)
        response.headers.set('Content-Type', 'image/png')
        return response
    else:
     
        return "Graph not found"

@app.route('/sample_status', methods=['GET', 'POST'])
def status_report():
    cur.execute("SELECT file_path,  DATE_FORMAT(time, '%H:%i') AS formatted_time, date, product_id FROM raw_data ORDER BY id DESC LIMIT 1")
    latest_raw_data = cur.fetchone()
    print("Latest raw data:", latest_raw_data)

    # Initialize default values
    file_path, formatted_time, file_date, product_id = ('None', 'N/A', 'N/A', None)

    if latest_raw_data:
        file_path, formatted_time, file_date, product_id = latest_raw_data

    # Initialize product_name and serial_no
    product_name, serial_no = 'Unknown Product', 'N/A'

    # Fetch product_name and serial_no from the item table using the product_id
    if product_id:
        cur.execute("SELECT product_name, serial_no FROM item WHERE product_id = %s", (product_id,))
        item_data = cur.fetchone()
        print("Item data:", item_data)
        if item_data:
            product_name, serial_no = item_data

    # Fetch test_id from the testing table for the specific product_id
    cur.execute("SELECT test_id FROM testing WHERE product_id = %s LIMIT 1", (product_id,))
    testing_data = cur.fetchone()
    print("Testing data:", testing_data)
    test_id = testing_data[0] if testing_data else 'N/A'

    # Fetch sample_status and overall_rms_value from the result_data table
    cur.execute("SELECT sample_status, overall_rms_value FROM result_data")
    result_data = cur.fetchone()
    print("Result data:", result_data)

    if result_data:
        sample_status, overall_rms_value = result_data
        overall_rms_value_list = overall_rms_value.split(',') if isinstance(overall_rms_value, str) else []

        # Change 'Bad Sample' to 'Not OK'
        if sample_status == 'Bad Sample':
            sample_status = 'NOK'
    else:
        sample_status = 'N/A'
        overall_rms_value_list = []
    return render_template('sample.html', product_name=product_name,
            serial_no=serial_no,
            test_id=test_id,
            file_path=file_path,
            sample_status=sample_status,
            overall_value=overall_rms_value_list,
            time=formatted_time,
            date=file_date)
@app.route('/dashboard')
def dashboard():
    try:
        # Fetch the latest entry from the raw_data table
        cur.execute("SELECT file_path, DATE_FORMAT(time, '%H:%i') AS formatted_time, date, product_id FROM raw_data ORDER BY id DESC LIMIT 1")
        latest_raw_data = cur.fetchone()
        print("Latest raw data:", latest_raw_data)

        # Initialize default values
        file_path, formatted_time, file_date, product_id = ('None', 'N/A', 'N/A', None)

        if latest_raw_data:
            file_path, formatted_time, file_date, product_id = latest_raw_data
        
        # Initialize product_name and serial_no
        product_name, serial_no = 'Unknown Product', 'N/A'

        # Fetch product_name and serial_no from the item table using the product_id
        if product_id:
            cur.execute("SELECT product_name, serial_no FROM item WHERE product_id = %s", (product_id,))
            item_data = cur.fetchone()
            print("Item data:", item_data)
            if item_data:
                product_name, serial_no = item_data

        # Fetch test_id from the testing table for the specific product_id
        cur.execute("SELECT test_id FROM testing WHERE product_id = %s LIMIT 1", (product_id,))
        testing_data = cur.fetchone()
        print("Testing data:", testing_data)
        test_id = testing_data[0] if testing_data else 'N/A'

        # Fetch sample_status and overall_rms_value from the result_data table for the specific product_id
        cur.execute("SELECT sample_status, overall_rms_value FROM result_data ")
        result_data = cur.fetchone()
        print("Result data:", result_data)

        # Check if result_data is None
        if result_data:
            sample_status, overall_rms_value = result_data
            overall_rms_value_list = overall_rms_value.split(',') if isinstance(overall_rms_value, str) else []
            
            # Change 'Bad Sample' to 'Not OK'
            if sample_status == 'Bad Sample':
                sample_status = 'Not OK'
        else:
            # Default values if no result is found
            sample_status = 'N/A'
            overall_rms_value_list = []

      # Count number of 'Bad Sample' (without filtering by product_id)
        # Count number of 'Bad Sample' (without filtering by product_id)
        cur.execute("SELECT COUNT(sample_status) FROM result_data WHERE sample_status='Bad Sample'")
        bad_sample_count = cur.fetchone()[0] or 0  # Use 0 if None
        print("Count of Bad Samples:", bad_sample_count)

        # Count total samples
        cur.execute("SELECT COUNT(sample_status) FROM result_data")
        total_sample_count = cur.fetchone()[0] or 0  # Use 0 if None
        print("Total Samples Count:", total_sample_count)

         # Prepare the res_items string
        res_items = f"{bad_sample_count}/{total_sample_count}" if total_sample_count > 0 else "0/0"
        print("Formatted res_items:", res_items)  # Debugging output


        # Pass the fetched data to the template
        return render_template(
            'dashboard.html',
            product_name=product_name,
            serial_no=serial_no,
            test_id=test_id,
            file_path=file_path,
            sample_status=sample_status,
            overall_value=overall_rms_value_list,
            time=formatted_time,
            date=file_date,
            res_items=res_items
        )

    except Exception as e:
        print(f'Error fetching data: {e}')
        return render_template('dashboard.html', error=str(e))

@socketio.on('connect')
def handle_connect():
    emit('progress_update', current_progress)
    emit('results_update', current_results)

@socketio.on('update_progress')
def handle_progress(data):
    global current_progress
    current_progress = data
    emit('progress', data, broadcast=True)
    
@socketio.on('update_results')
def handle_results(data):
    global current_results
    current_results = data
    try:
        overall_rms_value_str = ','.join(map(str, data['overall_rms_value']))

        # Create a temporary table to store the ID of the record to be updated
        cur.execute("CREATE TEMPORARY TABLE temp_table (id INT)")
        cur.execute("INSERT INTO temp_table (id) SELECT id FROM result_data ORDER BY timestamp DESC LIMIT 1")
        
        # Fetch the ID from the temporary table
        cur.execute("SELECT id FROM temp_table")
        record_id = cur.fetchone()[0]
        
        # Update the existing record in the database
        cur.execute(
            """
            UPDATE result_data
            SET sample_status = %s, overall_rms_value = %s, upper_limit = %s, lower_limit = %s, timestamp = NOW()
            WHERE id = %s
            """,
            (data['sample_status'], overall_rms_value_str, data['upper_limit'], data['lower_limit'], record_id)
        )
        conn.commit()
        
        # Drop the temporary table
        cur.execute("DROP TEMPORARY TABLE temp_table")

        # Emit the updated results to all connected clients
        emit('results', {
            'sample_status': data['sample_status'],
            'overall_rms_value': data['overall_rms_value'],
            'upper_limit': data['upper_limit'],
            'lower_limit': data['lower_limit'],
        }, broadcast=True)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    except Exception as e:
        print(f'Error emitting results: {e}')

if __name__ == '__main__':
    socketio.run(app, debug=True)