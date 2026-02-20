@echo off
echo ========================================
echo   PLANT HEALTH MONITORING SYSTEM
echo ========================================
echo.

cd /d "C:\Personal\start-up\Projects\PLANT_HEALTH_PROJECT\PlantHealthProject_pieso\code\esp32_plant_monitor"

echo [1/3] Checking Python environment...
"..\..\venv\Scripts\python.exe" --version

echo.
echo [2/3] Creating training data...
"..\..\venv\Scripts\python.exe" -c "
print('Generating plant health data...')
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Create directories
os.makedirs('../../data', exist_ok=True)
os.makedirs('../../models', exist_ok=True)

# Generate data
n = 1000
timestamps = [datetime.now() - timedelta(seconds=i) for i in range(n)]
vibrations = []
statuses = []

for i in range(n):
    if i < 400:
        v = np.random.normal(0.5, 0.1)
        s = 'healthy'
    elif i < 700:
        v = np.random.normal(1.0, 0.2)
        s = 'stressed'
    else:
        v = np.random.normal(2.0, 0.3)
        s = 'diseased'
    
    vibrations.append(max(0.1, v + np.random.normal(0, 0.05)))
    statuses.append(s)

df = pd.DataFrame({'timestamp': timestamps, 'vibration': vibrations, 'health_status': statuses})
df.to_csv('../../data/plant_vibration_data.csv', index=False)
print(f'Created {len(df)} samples')
print('Saved to: ../../data/plant_vibration_data.csv')
"

echo.
echo [3/3] Training AI model...
"..\..\venv\Scripts\python.exe" -c "
print('Training plant health model...')
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('../../data/plant_vibration_data.csv')
print(f'Loaded {len(df)} samples')

# Prepare
X = df[['vibration']].values
y = df['health_status'].values

# Encode
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
accuracy = model.score(X_test_scaled, y_test)
print(f'Model accuracy: {accuracy:.2%}')

# Save
model_data = {'model': model, 'scaler': scaler, 'le': le, 'accuracy': accuracy}
with open('../../models/plant_ai_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
with open('../../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Model saved to ../../models/')
"

echo.
echo [4/3] Launching Web Dashboard...
echo Open your browser to: http://localhost:5000
echo.
start http://localhost:5000
"..\..\venv\Scripts\python.exe" -c "
from flask import Flask, jsonify, render_template_string
import random
from datetime import datetime
import pickle
import numpy as np

app = Flask(__name__)

# Try load model
try:
    with open('../../models/plant_ai_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    le = model_data['le']
    MODEL = True
except:
    MODEL = False

html = '''
<!DOCTYPE html><html><head><title>Plant Health</title>
<style>body{font-family:Arial;margin:40px;background:#f5f5f5;}
.container{max-width:600px;margin:auto;background:white;padding:30px;border-radius:10px;box-shadow:0 0 20px rgba(0,0,0,0.1);}
.status{font-size:32px;font-weight:bold;margin:20px 0;padding:15px;border-radius:8px;text-align:center;}
.healthy{background:#d4edda;color:#155724;}
.stressed{background:#fff3cd;color:#856404;}
.diseased{background:#f8d7da;color:#721c24;}
.metric{display:flex;justify-content:space-between;margin:10px 0;padding:10px;background:#f8f9fa;border-radius:5px;}
</style></head>
<body><div class='container'>
<h1>?? Plant Health Monitor</h1>
<div class='status' id='status'>Loading...</div>
<div class='metric'><span>Vibration</span><span id='vib'>-</span></div>
<div class='metric'><span>Confidence</span><span id='conf'>-</span></div>
<div class='metric'><span>Time</span><span id='time'>-</span></div>
<button onclick='refresh()' style='margin-top:20px;padding:10px 20px;background:#007bff;color:white;border:none;border-radius:5px;cursor:pointer;'>?? Refresh</button>
</div>
<script>
function refresh(){fetch('/data').then(r=>r.json()).then(d=>{
document.getElementById('status').textContent=d.status;
document.getElementById('status').className='status '+d.status;
document.getElementById('vib').textContent=d.vibration.toFixed(3)+' V';
document.getElementById('conf').textContent=(d.confidence*100).toFixed(1)+'%';
document.getElementById('time').textContent=new Date(d.time).toLocaleTimeString();
});}
setInterval(refresh,3000);refresh();
</script></body></html>
'''

@app.route('/')
def home():
    return render_template_string(html)

@app.route('/data')
def data():
    vib = random.uniform(0.1, 3.0)
    if MODEL:
        try:
            scaled = scaler.transform([[vib]])
            pred = model.predict(scaled)[0]
            conf = np.max(model.predict_proba(scaled)[0])
            status = le.inverse_transform([pred])[0]
        except:
            if vib < 0.8: status='healthy';conf=0.9
            elif vib < 1.5: status='stressed';conf=0.8
            else: status='diseased';conf=0.7
    else:
        if vib < 0.8: status='healthy';conf=random.uniform(0.8,0.95)
        elif vib < 1.5: status='stressed';conf=random.uniform(0.7,0.9)
        else: status='diseased';conf=random.uniform(0.6,0.85)
    
    return jsonify({'status':status,'vibration':vib,'confidence':conf,'time':datetime.now().isoformat()})

if __name__ == '__main__':
    print('Dashboard: http://localhost:5000')
    app.run(debug=True, port=5000)
"

pause
