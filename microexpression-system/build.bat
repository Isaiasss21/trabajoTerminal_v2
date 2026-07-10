@echo off
REM build.bat — construye MicroExpressionAnalyzer.exe con PyInstaller
REM Ejecutar desde: microexpression-system/

echo Instalando PyInstaller y hooks...
pip install pyinstaller pyinstaller-hooks-contrib

echo.
echo Construyendo EXE (esto puede tardar 5-15 minutos)...
pyinstaller microexpression.spec --clean --noconfirm

echo.
echo Creando carpetas necesarias en dist/...
if not exist "dist\MicroExpressionAnalyzer\models" mkdir "dist\MicroExpressionAnalyzer\models"
if not exist "dist\MicroExpressionAnalyzer\data\sessions" mkdir "dist\MicroExpressionAnalyzer\data\sessions"

echo.
echo ===================================================
echo  Listo. El ejecutable esta en:
echo  dist\MicroExpressionAnalyzer\MicroExpressionAnalyzer.exe
echo.
echo  Copia tus modelos .pth a:
echo  dist\MicroExpressionAnalyzer\models\
echo ===================================================
pause
