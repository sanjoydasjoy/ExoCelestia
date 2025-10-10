@echo off
REM Authentication System Setup Script for Windows

echo 🚀 Setting up Authentication System...
echo.

REM Backend setup
echo 📦 Installing backend dependencies...
cd backend
pip install python-jose[cryptography]==3.3.0 passlib[bcrypt]==1.7.4 pydantic-settings==2.1.0
echo ✅ Backend dependencies installed!
echo.

REM Return to root
cd ..

echo ✨ Authentication system is ready!
echo.
echo 📚 Next steps:
echo 1. Start backend: cd backend ^&^& uvicorn app.main:app --reload
echo 2. Start frontend: cd frontend ^&^& npm run dev
echo 3. Visit http://localhost:3000 and try the Login/Sign Up buttons!
echo.
echo 📖 See AUTH_IMPLEMENTATION.md for full documentation
pause

