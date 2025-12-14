@echo off
echo ===============================
echo  PROJECT INSTALLER (Windows)
echo ===============================

REM ---- NPM INSTALL ----
IF EXIST package.json (
    echo 🔧 Menjalankan npm install ...
    npm install
) ELSE (
    echo ⚠️ package.json tidak ditemukan, skip npm install
)

REM ---- PYTHON VENV ----
IF NOT EXIST venv (
    echo 🐍 Membuat virtual environment ...
    python -m venv venv
) ELSE (
    echo ✔ venv sudah ada, skip pembuatan
)

echo 🔌 Mengaktifkan virtual environment ...
call venv\Scripts\activate

REM ---- PIP INSTALL ----
IF EXIST requirements.txt (
    echo 📦 Menjalankan pip install -r requirements.txt ...
    pip install -r requirements.txt
) ELSE (
    echo ⚠️ requirements.txt tidak ditemukan, skip pip install
)

echo ===============================
echo Instalasi selesai!
echo ===============================
pause
