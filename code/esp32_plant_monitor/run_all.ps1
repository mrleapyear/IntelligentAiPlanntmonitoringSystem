cd "C:\Personal\start-up\Projects\PLANT_HEALTH_PROJECT\PlantHealthProject_pieso\code\esp32_plant_monitor"
$py = "..\..\venv\Scripts\python.exe"

Write-Host "=== PLANT HEALTH PROJECT ===" -ForegroundColor Cyan
Write-Host "1. Creating data..." -ForegroundColor Yellow
& $py create_pvdf_data.py

Write-Host "`n2. Training model..." -ForegroundColor Yellow
& $py train_pvdf_model.py

Write-Host "`n3. Launching dashboard..." -ForegroundColor Yellow
Write-Host "Open: http://localhost:5000" -ForegroundColor Green
Start-Process "http://localhost:5000"
& $py web_dashboard.py
